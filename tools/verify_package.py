#!/usr/bin/env python3
"""Fast, offline integrity check for the code-review package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "benchmark/artifacts/folds.csv": "0470a4bb4630061ef06e92c50a74cb15ba457e476714294872ec426239a1ee6f",
    "artifacts/frozen_scores/anchor/fused_scores.pt": "5eadacd35f66732cd69a785a998d2b444703fc163a44e3842897df001f7c8e40",
    "artifacts/frozen_scores/complementary/F/component_scores.pt": "5d7af8abdc8ddc3a44bb3e8154526f44e73e01ecfe3b722294af0bcd7ad4dc66",
    "artifacts/frozen_scores/complementary/K/component_scores.pt": "bc3b6758f39de8f6271706ab88a1f6dfb153e04d43790d2c4e2eb8a1ddf8a7e7",
    "artifacts/final_result/submission.csv": "a37db3534c9476228799dd5acc8ee749cd21040980235f2fc2880820d632d678",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"sha256: {relative}")

    manifest = pd.read_csv(ROOT / "benchmark/artifacts/folds.csv")
    if (
        len(manifest) != 4067
        or manifest.sample_index.nunique() != 4067
        or manifest.individual_id.nunique() != 255
        or manifest.source_group.groupby(manifest.fold).nunique().sum() <= 0
        or manifest.image_path.astype(str).str.startswith("test/").any()
    ):
        failures.append("fixed manifest geometry")
    crossing = manifest.groupby("source_group").fold.nunique().gt(1).sum()
    if int(crossing) != 0:
        failures.append("source-group leakage")

    reference = None
    score_paths = [
        ROOT / "artifacts/frozen_scores/anchor/fused_scores.pt",
        ROOT / "artifacts/frozen_scores/complementary/F/component_scores.pt",
        ROOT / "artifacts/frozen_scores/complementary/K/component_scores.pt",
    ]
    for path in score_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        values = (
            np.asarray(payload["sample_index"]),
            np.asarray(payload["part_index"]),
            [str(x) for x in payload["label_order"]],
        )
        score = np.asarray(payload["classifier_score"])
        if score.shape != (1641, 255) or not np.isfinite(score).all():
            failures.append(f"score geometry: {path.relative_to(ROOT)}")
        if reference is None:
            reference = values
        elif not (
            np.array_equal(values[0], reference[0])
            and np.array_equal(values[1], reference[1])
            and values[2] == reference[2]
        ):
            failures.append(f"score alignment: {path.relative_to(ROOT)}")

    oversized = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size >= 1_000_000_000
    ]
    if oversized:
        failures.append(f"individual files >=1GB: {oversized}")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "package_bytes": sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file()),
        "manifest_rows": len(manifest),
        "identities": int(manifest.individual_id.nunique()),
        "source_group_crossing": int(crossing),
        "frozen_score_files": len(score_paths),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
