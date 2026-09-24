#!/usr/bin/env python3
"""Restore the organizer public ZIP into the canonical competition layout."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo


EXPECTED_PUBLIC_SHA256 = (
    "ec67b0977336fc5bd04f0c0cf2a5bdcd8c6071ecb2b2b060562e2c54aef27961"
)
EXPECTED_TEMPLATE_SHA256 = (
    "274c7db5bbc34763e1081dcd21b88abf0e087b761d13beb8b35fdc464d76f2f8"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_name(info: ZipInfo) -> str:
    name = info.filename
    if info.flag_bits & 0x800:
        return name
    try:
        return name.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def canonical_relative(name: str) -> PurePosixPath | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    parts = path.parts
    if not parts:
        return None
    if parts[0] == "数据集构建脚本" and len(parts) > 1:
        return PurePosixPath(*parts[1:])
    if parts[0] == "测试集" and len(parts) > 1:
        return PurePosixPath("test", *parts[1:])
    if parts[0] == "数据集划分" and len(parts) > 1:
        return PurePosixPath("splits", *parts[1:])
    if parts == ("submission_template.csv",):
        return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-zip", type=Path, required=True)
    parser.add_argument("--competition-root", type=Path, required=True)
    args = parser.parse_args()
    archive = args.public_zip.resolve()
    output_root = args.competition_root.resolve()
    actual_sha = sha256_file(archive)
    if actual_sha != EXPECTED_PUBLIC_SHA256:
        raise AssertionError(f"Unexpected public ZIP SHA-256: {actual_sha}")
    output_root.mkdir(parents=True, exist_ok=True)
    written = 0
    with ZipFile(archive) as source:
        for info in source.infolist():
            relative = canonical_relative(decoded_name(info))
            if relative is None:
                continue
            target = output_root.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as input_handle, target.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle)
            written += 1
    xml_count = sum(1 for _ in (output_root / "scripts/hyena_xml").glob("*.xml"))
    test_count = sum(1 for _ in (output_root / "test").glob("*Test/*.jpg"))
    template = output_root / "submission_template.csv"
    if xml_count != 2180 or test_count != 1641:
        raise AssertionError(
            f"Unexpected restored geometry: xml={xml_count}, test={test_count}"
        )
    if sha256_file(template) != EXPECTED_TEMPLATE_SHA256:
        raise AssertionError("Unexpected submission template SHA-256")
    print(
        f"restored_files={written} xml={xml_count} anonymous_test={test_count} "
        f"root={output_root}"
    )


if __name__ == "__main__":
    main()
