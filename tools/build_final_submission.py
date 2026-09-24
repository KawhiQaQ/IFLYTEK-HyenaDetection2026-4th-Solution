#!/usr/bin/env python3
"""Build the final non-TTA submission from aligned component raw scores.

The implementation consumes the six-component anchor score plus the F and K
complementary scores and applies the single pre-registered extrapolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPECTED_ROWS = 1641
EXPECTED_CLASSES = 255
EXPECTED_SUBMISSION_CSV_SHA256 = (
    "a37db3534c9476228799dd5acc8ee749cd21040980235f2fc2880820d632d678"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_score(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    score = np.asarray(payload["classifier_score"], dtype=np.float32)
    sample = np.asarray(payload["sample_index"], dtype=np.int64)
    part = np.asarray(payload["part_index"], dtype=np.int64)
    labels = [str(value) for value in payload["label_order"]]
    if score.shape != (EXPECTED_ROWS, EXPECTED_CLASSES):
        raise AssertionError(f"Unexpected score shape for {path}: {score.shape}")
    if not np.isfinite(score).all():
        raise AssertionError(f"Non-finite score in {path}")
    if not np.array_equal(sample, np.arange(EXPECTED_ROWS)):
        raise AssertionError(f"Unexpected sample order in {path}")
    if len(labels) != EXPECTED_CLASSES or len(set(labels)) != EXPECTED_CLASSES:
        raise AssertionError(f"Unexpected label order in {path}")
    return {"score": score, "sample": sample, "part": part, "labels": labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor-score",
        "--v10-score",
        dest="anchor_score",
        type=Path,
        required=True,
    )
    parser.add_argument("--f-score", type=Path, required=True)
    parser.add_argument("--k-score", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument(
        "--test-predictions",
        type=Path,
        required=True,
        help="Anchor test_predictions.csv; used only for the organizer part column/order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-reference-hash",
        action="store_true",
        help="Require the exact submitted CSV hash (for the bundled frozen-score audit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {args.output_dir}")

    anchor = load_score(args.anchor_score)
    component_f = load_score(args.f_score)
    component_k = load_score(args.k_score)
    for name, payload in (("F", component_f), ("K", component_k)):
        if (
            payload["labels"] != anchor["labels"]
            or not np.array_equal(payload["sample"], anchor["sample"])
            or not np.array_equal(payload["part"], anchor["part"])
        ):
            raise AssertionError(f"{name} score alignment differs from anchor")

    template = pd.read_csv(args.template)
    observations = pd.read_csv(args.test_predictions).sort_values(
        "sample_index"
    ).reset_index(drop=True)
    if list(template.columns) != ["image_id", "predicted_id"]:
        raise AssertionError("Unexpected organizer template columns")
    if len(template) != EXPECTED_ROWS or len(observations) != EXPECTED_ROWS:
        raise AssertionError("Unexpected test row count")
    if template.image_id.astype(str).tolist() != observations.image_id.astype(str).tolist():
        raise AssertionError("Template and test observation order differ")
    parts = observations.part.astype(str).to_numpy()
    if set(parts) != {"head", "left_body", "right_body"}:
        raise AssertionError("Unexpected organizer part values")

    # A residual extrapolation away from the complementary components.
    final = np.empty_like(anchor["score"])
    head = parts == "head"
    final[head] = (
        np.float32(1.30) * anchor["score"][head]
        - np.float32(0.30) * component_k["score"][head]
    )
    final[~head] = (
        np.float32(1.15) * anchor["score"][~head]
        - np.float32(0.15) * component_f["score"][~head]
    )
    if not np.isfinite(final).all():
        raise AssertionError("Final score contains non-finite values")

    labels = np.asarray(anchor["labels"], dtype=object)
    predicted = labels[final.argmax(axis=1)].astype(str)
    if not set(predicted).issubset(set(labels.astype(str))):
        raise AssertionError("Prediction outside the official 255-ID space")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    score_path = args.output_dir / "final_scores.pt"
    torch.save(
        {
            "version": "final_non_tta_ensemble",
            "sample_index": torch.from_numpy(anchor["sample"]),
            "part_index": torch.from_numpy(anchor["part"]),
            "label_order": list(anchor["labels"]),
            "classifier_score": torch.from_numpy(final),
            "formula": "head=1.30*anchor-0.30*K; body=1.15*anchor-0.15*F",
            "tta": False,
        },
        score_path,
    )
    submission = template[["image_id"]].copy()
    submission["predicted_id"] = predicted
    csv_path = args.output_dir / "submission.csv"
    zip_path = args.output_dir / "submission.zip"
    submission.to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["submission.csv"]:
            raise AssertionError("Submission ZIP must contain only submission.csv")
        archived = pd.read_csv(archive.open("submission.csv"))
    if not archived.equals(submission):
        raise AssertionError("ZIP content differs from submission.csv")

    csv_hash = sha256(csv_path)
    if args.require_reference_hash and csv_hash != EXPECTED_SUBMISSION_CSV_SHA256:
        raise AssertionError(
            f"Reference CSV mismatch: {csv_hash} != {EXPECTED_SUBMISSION_CSV_SHA256}"
        )
    report = {
        "status": "PASS",
        "method": "part-aware non-TTA raw-score extrapolation",
        "formula": {
            "head": "1.30*anchor-0.30*K",
            "left_body": "1.15*anchor-0.15*F",
            "right_body": "1.15*anchor-0.15*F",
        },
        "train_rows_per_component": 4067,
        "test_rows": EXPECTED_ROWS,
        "classes": EXPECTED_CLASSES,
        "tta": False,
        "test_labels_or_identity_used": False,
        "input_sha256": {
            "anchor_score": sha256(args.anchor_score),
            "f_score": sha256(args.f_score),
            "k_score": sha256(args.k_score),
            "template": sha256(args.template),
            "test_predictions": sha256(args.test_predictions),
        },
        "output_sha256": {
            "final_scores": sha256(score_path),
            "submission_csv": csv_hash,
            "submission_zip": sha256(zip_path),
        },
        "reference_csv_match": csv_hash == EXPECTED_SUBMISSION_CSV_SHA256,
    }
    report_path = args.output_dir / "reproduction_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
