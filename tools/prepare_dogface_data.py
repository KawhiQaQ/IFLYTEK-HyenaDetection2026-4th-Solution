from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image


EXPECTED_ARCHIVE_BYTES = 75_571_850
EXPECTED_ARCHIVE_MD5 = "010c207e202bb499039452aa7363b015"
EXPECTED_IMAGES = 8_363
EXPECTED_IDENTITIES = 1_393
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def decoded_pixel_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    header = f"RGB:{rgb.width}x{rgb.height}:".encode("ascii")
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()


def difference_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    packed = np.packbits(bits.reshape(-1)).tobytes()
    return packed.hex()


def official_pixel_hashes(root: Path) -> tuple[set[str], dict[str, int]]:
    patterns = (
        root / "hyena" / "cropped_images",
        root / "test",
    )
    hashes: set[str] = set()
    counts: dict[str, int] = {}
    for source in patterns:
        count = 0
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                with Image.open(path) as image:
                    image.load()
                    hashes.add(decoded_pixel_digest(image))
                count += 1
        counts[str(source)] = count
    return hashes, counts


def image_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise AssertionError(f"Unsafe archive member: {info.filename}")
        if info.is_dir() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if len(path.parts) < 2:
            raise AssertionError(f"Image has no identity directory: {info.filename}")
        members.append(info)
    return members


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path = args.archive.resolve()
    output_root = args.output_root.resolve()
    competition_root = args.competition_root.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise AssertionError(
            f"Archive bytes {archive_path.stat().st_size} != {EXPECTED_ARCHIVE_BYTES}"
        )
    archive_md5 = digest_file(archive_path, "md5")
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise AssertionError(f"Archive MD5 changed: {archive_md5}")

    images_root = output_root / "images"
    manifest_path = output_root / "manifest.csv"
    audit_path = output_root / "audit.json"
    if (images_root.exists() or manifest_path.exists() or audit_path.exists()) and not args.force:
        raise FileExistsError(f"Prepared output already exists: {output_root}")
    if args.force and images_root.exists():
        shutil.rmtree(images_root)
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    official_hashes, official_counts = official_pixel_hashes(competition_root)
    records: list[dict[str, object]] = []
    byte_hash_labels: dict[str, set[str]] = defaultdict(set)
    pixel_hash_labels: dict[str, set[str]] = defaultdict(set)
    dimensions: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        members = image_members(archive)
        for info in members:
            source_path = PurePosixPath(info.filename)
            identity = source_path.parent.name
            payload = archive.read(info)
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                width, height = image.size
                if image.mode not in {"RGB", "RGBA", "L"}:
                    raise AssertionError(
                        f"Unexpected mode {image.mode}: {info.filename}"
                    )
                pixel_sha256 = decoded_pixel_digest(image)
                dhash = difference_hash(image)
            dimensions[f"{width}x{height}"] += 1
            byte_sha256 = digest_bytes(payload)
            byte_hash_labels[byte_sha256].add(identity)
            pixel_hash_labels[pixel_sha256].add(identity)
            target_dir = images_root / identity
            target_dir.mkdir(parents=True, exist_ok=True)
            target_name = source_path.name
            target = target_dir / target_name
            if target.exists():
                target = target_dir / f"{source_path.stem}_{byte_sha256[:10]}{source_path.suffix.lower()}"
            target.write_bytes(payload)
            records.append(
                {
                    "external_dataset": "DogFaceNet_224resized",
                    "identity_name": identity,
                    "image_path": target.relative_to(output_root).as_posix(),
                    "archive_member": info.filename,
                    "archive_bytes_sha256": byte_sha256,
                    "decoded_pixel_sha256": pixel_sha256,
                    "dhash64": dhash,
                    "width": width,
                    "height": height,
                    "competition_exact_pixel_overlap": pixel_sha256 in official_hashes,
                }
            )

    frame = pd.DataFrame.from_records(records)
    identities = sorted(frame.identity_name.astype(str).unique())
    label_lookup = {identity: index for index, identity in enumerate(identities)}
    frame.insert(0, "sample_index", np.arange(len(frame), dtype=np.int64))
    frame.insert(3, "label_index", frame.identity_name.map(label_lookup).astype(int))
    frame.insert(4, "part", "head")
    frame.insert(5, "part_index", 0)
    frame.insert(6, "source_code", frame.sample_index.astype(int))
    # DogFaceNet publishes aligned crops rather than encounter/source groups.
    # Treat every released image as a distinct source so P x K sampling never
    # invents a same-source relationship between different face photographs.
    frame.insert(7, "source_group", frame.sample_index.map(lambda value: f"dogface_{value}"))
    frame = frame.sort_values(["label_index", "image_path"]).reset_index(drop=True)
    frame.sample_index = np.arange(len(frame), dtype=np.int64)
    frame.source_code = frame.sample_index.astype(int)
    frame.source_group = frame.sample_index.map(lambda value: f"dogface_{value}")

    cross_identity_byte_conflicts = {
        key: sorted(value)
        for key, value in byte_hash_labels.items()
        if len(value) > 1
    }
    cross_identity_pixel_conflicts = {
        key: sorted(value)
        for key, value in pixel_hash_labels.items()
        if len(value) > 1
    }
    overlap_count = int(frame.competition_exact_pixel_overlap.sum())
    status = "PASS"
    failures: list[str] = []
    if len(frame) != EXPECTED_IMAGES:
        failures.append(f"image_count={len(frame)}")
    if len(identities) != EXPECTED_IDENTITIES:
        failures.append(f"identity_count={len(identities)}")
    if overlap_count:
        failures.append(f"competition_exact_pixel_overlap={overlap_count}")
    if cross_identity_byte_conflicts:
        failures.append(
            f"cross_identity_byte_conflicts={len(cross_identity_byte_conflicts)}"
        )
    if cross_identity_pixel_conflicts:
        failures.append(
            f"cross_identity_pixel_conflicts={len(cross_identity_pixel_conflicts)}"
        )
    if failures:
        status = "FAIL"

    frame.to_csv(manifest_path, index=False)
    audit = {
        "status": status,
        "failures": failures,
        "dataset": "DogFaceNet_224resized",
        "declared_species": ["dog"],
        "hyena_or_hyenaid_present": False,
        "wildbook_or_competition_source_present": False,
        "license": "CC-BY-4.0",
        "source_record": "https://zenodo.org/records/12578449",
        "source_api": "https://zenodo.org/api/records/12578449",
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_md5": archive_md5,
        "archive_sha256": digest_file(archive_path),
        "images": int(len(frame)),
        "identities": int(len(identities)),
        "min_images_per_identity": int(frame.groupby("label_index").size().min()),
        "max_images_per_identity": int(frame.groupby("label_index").size().max()),
        "dimensions": dict(sorted(dimensions.items())),
        "cross_identity_byte_conflicts": len(cross_identity_byte_conflicts),
        "cross_identity_pixel_conflicts": len(cross_identity_pixel_conflicts),
        "competition_hash_audit": {
            "method": "decoded RGB pixel SHA-256 only; no label or feature enters training",
            "official_paths_and_counts": official_counts,
            "official_unique_pixel_hashes": len(official_hashes),
            "exact_pixel_overlap": overlap_count,
        },
        "manifest": str(manifest_path),
        "manifest_sha256": digest_file(manifest_path),
        "training_path_prefix": "images/",
        "anonymous_test_rows_in_manifest": 0,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
