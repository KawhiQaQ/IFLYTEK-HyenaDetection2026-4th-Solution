from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PARTS = ("head", "left_body", "right_body")


def competition_metrics(
    frame: pd.DataFrame,
    true_col: str = "individual_id",
    pred_col: str = "predicted_id",
    part_col: str = "part",
) -> dict[str, Any]:
    """Compute the public competition metric on a prediction frame."""
    required = {true_col, pred_col, part_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")

    per_part: dict[str, dict[str, float | int]] = {}
    for part in PARTS:
        subset = frame.loc[frame[part_col] == part]
        if subset.empty:
            raise ValueError(f"No rows found for required part: {part}")
        y_true = subset[true_col].astype(str)
        y_pred = subset[pred_col].astype(str)
        per_part[part] = {
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "top1_accuracy": float(accuracy_score(y_true, y_pred)),
            "n_samples": int(len(subset)),
            "n_true_ids": int(y_true.nunique()),
            "n_predicted_ids": int(y_pred.nunique()),
        }

    part_f1s = [float(per_part[p]["macro_f1"]) for p in PARTS]
    part_accs = [float(per_part[p]["top1_accuracy"]) for p in PARTS]
    return {
        "final_score": float(np.mean(part_f1s)),
        "mean_part_top1_accuracy": float(np.mean(part_accs)),
        "min_part_macro_f1": float(np.min(part_f1s)),
        "sample_weighted_top1_accuracy": float(
            accuracy_score(frame[true_col].astype(str), frame[pred_col].astype(str))
        ),
        "n_samples": int(len(frame)),
        "per_part": per_part,
    }

