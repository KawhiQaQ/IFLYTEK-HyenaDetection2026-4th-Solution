#!/usr/bin/env python3
"""Independent reconstruction audit of the six-model anchor."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_full import make_test_frame, sha256_file
from train_full_v4_2 import EXPECTED_TEMPLATE_SHA256


WEIGHTS = {
    "head": {"Q": 0.50, "H": 0.20, "D": 0.10, "C": 0.20},
    "left_body": {"Q": 0.25, "A": 0.5625, "B": 0.1875},
    "right_body": {"Q": 0.75, "A": 0.25},
}
EPOCHS = {"H": 28, "A": 37, "B": 36, "Q": 32, "D": 33, "C": 24}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for key, value in vars(args).items():
        setattr(args, key, value.resolve())
    manifest = pd.read_csv(args.manifest)
    if sha256_file(args.manifest) != "0470a4bb4630061ef06e92c50a74cb15ba457e476714294872ec426239a1ee6f":
        raise AssertionError("manifest hash mismatch")
    if len(manifest) != 4067 or manifest.image_path.astype(str).str.startswith("test/").any():
        raise AssertionError("training boundary mismatch")
    if sha256_file(args.template) != EXPECTED_TEMPLATE_SHA256:
        raise AssertionError("template hash mismatch")
    template = pd.read_csv(args.template)
    test_frame = make_test_frame(template)

    labels = None
    sample = None
    part = None
    scores = {}
    lifecycle = {}
    for key, epochs in EPOCHS.items():
        directory = args.component_root / key
        config = json.loads((directory / "config.json").read_text())
        report = json.loads((directory / "component_report.json").read_text())
        cleanup = json.loads((directory / "checkpoint_cleanup.json").read_text())
        log = [json.loads(line) for line in (directory / "train_log.jsonl").read_text().splitlines() if line]
        if [int(row["epoch"]) for row in log] != list(range(epochs)):
            raise AssertionError(f"non-contiguous epochs: {key}")
        if (
            config.get("init_checkpoint") is not None
            or config.get("competition_checkpoint_loaded") is not False
            or int(config.get("train_samples", -1)) != 4067
            or int(config.get("train_epochs", -1)) != epochs
            or config.get("tta_flip") is not False
            or report.get("checkpoint_retained") is not False
            or cleanup.get("deleted_checkpoint_sha256") != report.get("checkpoint_sha256")
            or (directory / "model.pt").exists()
        ):
            raise AssertionError(f"component lifecycle/provenance mismatch: {key}")
        payload = torch.load(directory / "component_scores.pt", map_location="cpu", weights_only=False)
        current_labels = [str(value) for value in payload["label_order"]]
        current_sample = np.asarray(payload["sample_index"], dtype=np.int64)
        current_part = np.asarray(payload["part_index"], dtype=np.int64)
        current_scores = np.asarray(payload["classifier_score"], dtype=np.float32)
        if current_scores.shape != (1641, 255) or not np.isfinite(current_scores).all():
            raise AssertionError(f"invalid scores: {key}")
        if labels is None:
            labels, sample, part = current_labels, current_sample, current_part
        elif current_labels != labels or not np.array_equal(current_sample, sample) or not np.array_equal(current_part, part):
            raise AssertionError(f"unaligned component: {key}")
        scores[key] = current_scores
        lifecycle[key] = {
            "epochs": epochs,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "scores_sha256": sha256_file(directory / "component_scores.pt"),
            "config_sha256": sha256_file(directory / "config.json"),
            "train_log_sha256": sha256_file(directory / "train_log.jsonl"),
        }
    assert labels is not None and sample is not None and part is not None
    if not np.array_equal(sample, np.arange(1641)) or not np.array_equal(part, test_frame.part_index.to_numpy()):
        raise AssertionError("test ordering mismatch")
    fused = np.zeros((1641, 255), dtype=np.float32)
    for part_name, weights in WEIGHTS.items():
        mask = test_frame.part.astype(str).to_numpy() == part_name
        for key, weight in weights.items():
            fused[mask] += np.float32(weight) * scores[key][mask]
    expected = np.asarray(labels, dtype=object)[fused.argmax(1)].astype(str)
    selected_scores = torch.load(args.output_dir / "fused_scores.pt", map_location="cpu", weights_only=False)
    if not np.array_equal(np.asarray(selected_scores["classifier_score"]), fused):
        raise AssertionError("fused score reconstruction mismatch")
    predictions = pd.read_csv(args.output_dir / "test_predictions.csv")
    submission = pd.read_csv(args.output_dir / "submission.csv")
    if not np.array_equal(predictions.predicted_id.astype(str).to_numpy(), expected):
        raise AssertionError("prediction reconstruction mismatch")
    if not np.array_equal(submission.predicted_id.astype(str).to_numpy(), expected):
        raise AssertionError("submission reconstruction mismatch")
    if submission.image_id.astype(str).tolist() != template.image_id.astype(str).tolist():
        raise AssertionError("template order mismatch")
    zip_path = args.output_dir / "anchor_submission.zip"
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["submission.csv"]:
            raise AssertionError("ZIP member mismatch")
        archived = pd.read_csv(archive.open("submission.csv"))
    if not archived.equals(submission):
        raise AssertionError("ZIP content mismatch")
    audit = {
        "status": "PASS",
        "version": "V10.3-full",
        "manifest_sha256": sha256_file(args.manifest),
        "template_sha256": sha256_file(args.template),
        "train_rows_per_component": 4067,
        "test_rows": 1641,
        "anonymous_test_train_rows": 0,
        "competition_checkpoint_initializers": 0,
        "tta_flip": False,
        "weights": WEIGHTS,
        "component_lifecycle": lifecycle,
        "zip_members": ["submission.csv"],
        "output_hashes": {
            "fused_scores": sha256_file(args.output_dir / "fused_scores.pt"),
            "test_predictions": sha256_file(args.output_dir / "test_predictions.csv"),
            "submission_csv": sha256_file(args.output_dir / "submission.csv"),
            "submission_zip": sha256_file(zip_path),
        },
    }
    audit_path = args.output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
