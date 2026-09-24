#!/usr/bin/env python3
"""Fuse the six aligned anchor-model score files."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_full import make_test_frame, sha256_file
from train_full_v4_2 import EXPECTED_TEMPLATE_SHA256
from train_anchor_component import COMPONENTS


WEIGHTS = {
    "head": {"Q": 0.50, "H": 0.20, "D": 0.10, "C": 0.20},
    "left_body": {"Q": 0.25, "A": 0.5625, "B": 0.1875},
    "right_body": {"Q": 0.75, "A": 0.25},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.component_root = args.component_root.resolve()
    args.template = args.template.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty final output: {args.output_dir}")
    if sha256_file(args.template) != EXPECTED_TEMPLATE_SHA256:
        raise AssertionError("template hash mismatch")
    template = pd.read_csv(args.template)
    test_frame = make_test_frame(template)
    labels: list[str] | None = None
    reference_sample: np.ndarray | None = None
    reference_part: np.ndarray | None = None
    scores: dict[str, np.ndarray] = {}
    component_hashes: dict[str, dict[str, str]] = {}
    for key, spec in COMPONENTS.items():
        directory = args.component_root / key
        score_path = directory / "component_scores.pt"
        report_path = directory / "component_report.json"
        cleanup_path = directory / "checkpoint_cleanup.json"
        report = json.loads(report_path.read_text())
        cleanup = json.loads(cleanup_path.read_text())
        if report.get("status") != "PASS" or cleanup.get("status") != "PASS":
            raise AssertionError(f"component {key} lifecycle did not pass")
        if int(report.get("train_epochs", -1)) != int(spec["epochs"]):
            raise AssertionError(f"component {key} epoch mismatch")
        if report.get("checkpoint_retained") is not False:
            raise AssertionError(f"component {key} checkpoint lifecycle incomplete")
        if (directory / "model.pt").exists():
            raise AssertionError(f"component {key} checkpoint unexpectedly retained")
        payload = torch.load(score_path, map_location="cpu", weights_only=False)
        if payload.get("component") != key:
            raise AssertionError(f"component score key mismatch: {key}")
        current_labels = [str(value) for value in payload["label_order"]]
        current_sample = np.asarray(payload["sample_index"], dtype=np.int64)
        current_part = np.asarray(payload["part_index"], dtype=np.int64)
        current_scores = np.asarray(payload["classifier_score"], dtype=np.float32)
        if current_scores.shape != (1641, 255) or not np.isfinite(current_scores).all():
            raise AssertionError(f"invalid score tensor: {key}")
        if labels is None:
            labels = current_labels
            reference_sample = current_sample
            reference_part = current_part
        elif (
            current_labels != labels
            or not np.array_equal(current_sample, reference_sample)
            or not np.array_equal(current_part, reference_part)
        ):
            raise AssertionError(f"component alignment mismatch: {key}")
        scores[key] = current_scores
        component_hashes[key] = {
            "scores": sha256_file(score_path),
            "report": sha256_file(report_path),
            "cleanup": sha256_file(cleanup_path),
        }
    assert labels is not None and reference_sample is not None and reference_part is not None
    if not np.array_equal(reference_sample, np.arange(1641)):
        raise AssertionError("final sample order changed")
    if not np.array_equal(reference_part, test_frame.part_index.to_numpy(dtype=np.int64)):
        raise AssertionError("final part order changed")

    fused = np.zeros((1641, 255), dtype=np.float32)
    for part_name, weights in WEIGHTS.items():
        mask = test_frame.part.astype(str).to_numpy() == part_name
        for key, weight in weights.items():
            fused[mask] += np.float32(weight) * scores[key][mask]
    predicted = np.asarray(labels, dtype=object)[fused.argmax(axis=1)].astype(str)
    if not set(predicted).issubset(set(labels)):
        raise AssertionError("prediction outside 255-label space")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "fused_scores.pt"
    torch.save(
        {
            "version": "V10.3-full",
            "sample_index": torch.from_numpy(reference_sample),
            "part_index": torch.from_numpy(reference_part),
            "label_order": labels,
            "classifier_score": torch.from_numpy(fused),
            "weights": WEIGHTS,
        },
        score_path,
    )
    pred_frame = test_frame.copy()
    pred_frame["predicted_id"] = predicted
    pred_frame["confidence"] = fused.max(axis=1)
    pred_path = args.output_dir / "test_predictions.csv"
    pred_frame.to_csv(pred_path, index=False)
    submission = template[["image_id"]].copy()
    submission["predicted_id"] = predicted
    csv_path = args.output_dir / "submission.csv"
    submission.to_csv(csv_path, index=False)
    zip_path = args.output_dir / "anchor_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["submission.csv"]:
            raise AssertionError("unexpected ZIP members")
        archived = pd.read_csv(archive.open("submission.csv"))
    if not archived.equals(submission):
        raise AssertionError("ZIP submission content differs")
    report = {
        "status": "PASS",
        "version": "V10.3-full",
        "train_rows_per_component": 4067,
        "test_rows": 1641,
        "tta_flip": False,
        "decoder": "part-aware raw-score convex fusion then argmax",
        "weights": WEIGHTS,
        "component_hashes": component_hashes,
        "template_sha256": sha256_file(args.template),
        "unique_predicted_ids": int(pd.Series(predicted).nunique()),
        "unique_predicted_ids_per_part": (
            pred_frame.groupby("part").predicted_id.nunique().astype(int).to_dict()
        ),
        "output_hashes": {
            "fused_scores": sha256_file(score_path),
            "test_predictions": sha256_file(pred_path),
            "submission_csv": sha256_file(csv_path),
            "submission_zip": sha256_file(zip_path),
        },
    }
    (args.output_dir / "inference_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
