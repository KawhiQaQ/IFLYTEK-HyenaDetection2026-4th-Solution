#!/usr/bin/env python3
"""Verify the external-weight bundle and seed an offline Hugging Face cache.

The weight bundle is intentionally distributed separately from the sub-1GB
code package.  This helper accepts the downloaded directory as-is, validates
every file used by the final pipeline, and exposes the two timm DINOv3 files
under the cache layout expected by ``huggingface_hub``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]

FILES: dict[str, dict[str, object]] = {
    "dinov3_huge_plus/model.safetensors": {
        "bytes": 3_362_084_520,
        "sha256": "0d54131c846c91a78ec7641482f32e45b27cb4895f83653c038f305f18c7e61c",
    },
    "dinov3_huge_plus/config.json": {
        "bytes": 675,
        "sha256": "c22d89c036033b9f22445c136e6e050a27dd7f3bff26566b9dad0c66c9c84a83",
    },
    "dinov3_convnext_large/model.safetensors": {
        "bytes": 784_955_728,
        "sha256": "77f63fee584d2c38416b865ee4b1bf9d4416df93a44ef641c2675de08d538a7b",
    },
    "dinov3_convnext_large/config.json": {
        "bytes": 596,
        "sha256": "c1c0a99e8311dce54b64ff0f85b0de034c7b9614eb3f56414fea4a0b1f7b176c",
    },
    "bioclip/open_clip_pytorch_model.bin": {
        "bytes": 598_599_013,
        "sha256": "e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4",
    },
    "bioclip/open_clip_config.json": {
        "bytes": 469,
        "sha256": "86d418d7046fa9212ab70eb7cb3deeef02be46f39d7948bad5dc7649f8277208",
    },
    "sam2/sam2.1_hiera_tiny.pt": {
        "bytes": 156_008_466,
        "sha256": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    },
    "dogface_stage_a/external_reid_final.pt": {
        "bytes": 3_487_139_153,
        "sha256": "1dfabf91e0b217f0f7dcf7c2881440a60f199a95868c0fc7c145293aa8876c69",
    },
    "dogface_stage_a/checkpoint_audit.json": {
        "bytes": 1_928,
        "sha256": "d6628e8dfa2d4ec0bb36a7592670588c21b7458a4e9cb0a8c88c22f918d6b6f2",
    },
}

HF_REPOS = {
    "timm/vit_huge_plus_patch16_dinov3.lvd1689m": {
        "commit": "dd0addc09788111fa893d24799a44744ba022eee",
        "source_dir": "dinov3_huge_plus",
    },
    "timm/convnext_large.dinov3_lvd1689m": {
        "commit": "0f8b496db5dea10c7533e85304400f0eddad55ac",
        "source_dir": "dinov3_convnext_large",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source, destination.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=CODE_ROOT.parent / "weights",
        help="Downloaded weights directory.",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        help="Cache destination; defaults to <weights-root>/hf_home.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate files without creating the offline cache links.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_root = args.weights_root.expanduser().resolve()
    hf_home = (
        args.hf_home.expanduser().resolve()
        if args.hf_home is not None
        else weights_root / "hf_home"
    )

    failures: list[str] = []
    verified: dict[str, dict[str, object]] = {}
    for relative, expected in FILES.items():
        path = weights_root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        verified[relative] = {
            "bytes": actual_bytes,
            "sha256": actual_sha256,
        }
        if actual_bytes != expected["bytes"]:
            failures.append(
                f"size: {relative}: {actual_bytes} != {expected['bytes']}"
            )
        if actual_sha256 != expected["sha256"]:
            failures.append(f"sha256: {relative}")

    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
        raise SystemExit(2)

    if not args.verify_only:
        hub_root = hf_home / "hub"
        for repo_id, spec in HF_REPOS.items():
            cache_name = "models--" + repo_id.replace("/", "--")
            repo_root = hub_root / cache_name
            commit = str(spec["commit"])
            source_dir = weights_root / str(spec["source_dir"])
            refs = repo_root / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text(commit, encoding="utf-8")
            snapshot = repo_root / "snapshots" / commit
            for filename in ("model.safetensors", "config.json"):
                relative_symlink(source_dir / filename, snapshot / filename)

    result = {
        "status": "PASS",
        "weights_root": str(weights_root),
        "verified_files": len(verified),
        "verified_bytes": sum(int(item["bytes"]) for item in verified.values()),
        "hf_home": None if args.verify_only else str(hf_home),
        "environment": {
            "HF_HOME": str(hf_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DOGFACE_STAGE_A_CHECKPOINT": str(
                weights_root / "dogface_stage_a/external_reid_final.pt"
            ),
            "DOGFACE_STAGE_A_AUDIT": str(
                weights_root / "dogface_stage_a/checkpoint_audit.json"
            ),
            "BIOCLIP_CHECKPOINT": str(
                weights_root / "bioclip/open_clip_pytorch_model.bin"
            ),
            "SAM2_CHECKPOINT": str(weights_root / "sam2/sam2.1_hiera_tiny.pt"),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
