from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metrics import competition_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    prediction_paths = sorted(Path(path) for path in glob.glob(args.predictions_glob))
    if not prediction_paths:
        raise FileNotFoundError(f"No predictions matched {args.predictions_glob}")
    predictions = pd.concat(
        [pd.read_csv(path) for path in prediction_paths], ignore_index=True
    )
    if "sample_index" not in predictions.columns:
        if "image_path" not in predictions.columns:
            raise ValueError(
                "Predictions require sample_index (preferred) or image_path"
            )
        path_to_index = manifest.set_index("image_path")["sample_index"]
        predictions["sample_index"] = predictions["image_path"].map(path_to_index)
        if predictions["sample_index"].isna().any():
            bad = predictions.loc[
                predictions["sample_index"].isna(), "image_path"
            ].tolist()[:5]
            raise AssertionError(f"Unknown image_path values: {bad}")
        predictions["sample_index"] = predictions["sample_index"].astype(int)
    if "predicted_id" not in predictions.columns:
        raise ValueError("Predictions require a predicted_id column")
    if predictions["sample_index"].duplicated().any():
        duplicates = predictions.loc[
            predictions["sample_index"].duplicated(), "sample_index"
        ].tolist()[:5]
        raise AssertionError(f"Duplicate OOF predictions: {duplicates}")

    truth = manifest.set_index("sample_index")
    indexed = predictions.set_index("sample_index")
    unknown = indexed.index.difference(truth.index)
    if len(unknown):
        raise AssertionError(f"Predictions contain unknown sample indices: {unknown[:5]}")
    expected = truth.loc[indexed.index].copy()
    label_space = set(manifest["individual_id"].astype(str))
    predicted_ids = indexed["predicted_id"].astype(str)
    invalid_ids = sorted(set(predicted_ids).difference(label_space))
    if invalid_ids:
        raise AssertionError(f"Predictions outside official label space: {invalid_ids[:5]}")
    evaluation = expected.reset_index()
    evaluation["predicted_id"] = predicted_ids.to_numpy()

    present_folds = sorted(evaluation["fold"].astype(int).unique().tolist())
    if args.require_complete:
        if present_folds != list(range(5)) or len(predictions) != len(manifest):
            raise AssertionError(
                f"Full OOF required, found folds={present_folds}, rows={len(predictions)}"
            )

    per_fold: list[dict[str, Any]] = []
    for fold in present_folds:
        fold_metrics = competition_metrics(evaluation.loc[evaluation["fold"] == fold])
        per_fold.append({"fold": fold, **fold_metrics})
    combined = competition_metrics(evaluation)
    fold_scores = [float(item["final_score"]) for item in per_fold]
    report = {
        "status": "full_oof" if present_folds == list(range(5)) else "partial_oof",
        "present_folds": present_folds,
        "prediction_files": [str(path) for path in prediction_paths],
        "coverage_rows": int(len(predictions)),
        "coverage_fraction": float(len(predictions) / len(manifest)),
        "mean_completed_fold_score": float(np.mean(fold_scores)),
        "std_completed_fold_score": float(np.std(fold_scores)),
        "combined_predictions_metrics": combined,
        "per_fold": per_fold,
    }
    if args.oof_output is not None:
        args.oof_output.parent.mkdir(parents=True, exist_ok=True)
        evaluation.sort_values("sample_index").to_csv(
            args.oof_output, index=False, encoding="utf-8"
        )
        report["oof_predictions"] = str(args.oof_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
