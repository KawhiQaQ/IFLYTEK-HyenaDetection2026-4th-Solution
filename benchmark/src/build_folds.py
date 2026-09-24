from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from metrics import PARTS


EXPECTED_COUNTS = {"head": 1963, "left_body": 1056, "right_body": 1048}
PART_FILES = {
    "head": "hyena_head_train.txt",
    "left_body": "hyena_left_body_train.txt",
    "right_body": "hyena_right_body_train.txt",
}
PART_SUFFIXES = {
    "head": "_head.jpg",
    "left_body": "_left_body.jpg",
    "right_body": "_right_body.jpg",
}


def parse_official_lists(competition_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for part_index, part in enumerate(PARTS):
        split_path = competition_root / "splits" / PART_FILES[part]
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        lines = split_path.read_text(encoding="utf-8-sig").splitlines()
        paths = [line.strip() for line in lines if line.strip()]
        if len(paths) != EXPECTED_COUNTS[part]:
            raise ValueError(
                f"{part}: expected {EXPECTED_COUNTS[part]} rows, got {len(paths)}"
            )
        suffix = PART_SUFFIXES[part]
        for relpath_text in paths:
            relpath = Path(relpath_text)
            if not relpath.name.endswith(suffix):
                raise ValueError(f"Unexpected {part} crop filename: {relpath.name}")
            individual_id = relpath.parent.name
            source_stem = relpath.name[: -len(suffix)]
            source_relpath = Path("hyena/hyena_images") / individual_id / (
                source_stem + ".jpg"
            )
            crop_path = competition_root / relpath
            source_path = competition_root / source_relpath
            if not crop_path.is_file():
                raise FileNotFoundError(crop_path)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            rows.append(
                {
                    "image_path": relpath.as_posix(),
                    "source_image": source_relpath.as_posix(),
                    "source_group": source_stem,
                    "part": part,
                    "part_index": part_index,
                    "individual_id": individual_id,
                    "stratum": f"{part}::{individual_id}",
                }
            )

    frame = pd.DataFrame(rows)
    if len(frame) != sum(EXPECTED_COUNTS.values()):
        raise AssertionError("Official total count mismatch")
    if frame["image_path"].duplicated().any():
        raise AssertionError("Duplicate crop path in official lists")
    group_id_counts = frame.groupby("source_group")["individual_id"].nunique()
    if int(group_id_counts.max()) != 1:
        bad = group_id_counts[group_id_counts > 1].index.tolist()[:5]
        raise AssertionError(f"A source image maps to multiple IDs: {bad}")

    labels = sorted(frame["individual_id"].unique())
    label_to_index = {label: index for index, label in enumerate(labels)}
    frame["label_index"] = frame["individual_id"].map(label_to_index).astype(int)
    frame.insert(0, "sample_index", np.arange(len(frame), dtype=np.int64))
    return frame


def make_candidate(frame: pd.DataFrame, seed: int, n_splits: int) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    folds = np.full(len(frame), -1, dtype=np.int16)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The least populated class in y has only"
        )
        for fold, (_, valid_indices) in enumerate(
            splitter.split(
                X=np.zeros(len(frame)),
                y=frame["stratum"],
                groups=frame["source_group"],
            )
        ):
            folds[valid_indices] = fold
    if np.any(folds < 0):
        raise AssertionError("Candidate did not assign every row")
    return folds


def _normalized_table_loss(
    values: pd.Series, folds: np.ndarray, n_splits: int
) -> tuple[float, float]:
    table = pd.crosstab(values, folds).reindex(columns=range(n_splits), fill_value=0)
    counts = table.to_numpy(dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    expected = totals / n_splits
    variance = np.mean(np.sum((counts - expected) ** 2, axis=1) / (totals[:, 0] ** 2 + 1.0))
    desired_presence = np.minimum(totals[:, 0], n_splits)
    actual_presence = (counts > 0).sum(axis=1)
    presence_loss = np.mean(
        (desired_presence - actual_presence) / np.maximum(desired_presence, 1.0)
    )
    return float(variance), float(presence_loss)


def candidate_objective(
    frame: pd.DataFrame, folds: np.ndarray, n_splits: int
) -> tuple[float, dict[str, float]]:
    stratum_variance, stratum_presence = _normalized_table_loss(
        frame["stratum"], folds, n_splits
    )
    id_variance, id_presence = _normalized_table_loss(
        frame["individual_id"], folds, n_splits
    )

    fold_sizes = np.bincount(folds, minlength=n_splits).astype(np.float64)
    size_cv2 = float(np.var(fold_sizes) / (np.mean(fold_sizes) ** 2 + 1.0))
    part_table = pd.crosstab(frame["part"], folds).reindex(
        index=PARTS, columns=range(n_splits), fill_value=0
    )
    part_counts = part_table.to_numpy(dtype=np.float64)
    part_means = part_counts.mean(axis=1, keepdims=True)
    part_cv2 = float(np.mean((part_counts - part_means) ** 2 / (part_means**2 + 1.0)))

    stratum_totals = frame["stratum"].value_counts()
    id_source_group_totals = frame.groupby("individual_id")["source_group"].nunique()
    avoidable_unsupported = 0
    unseen_global = 0
    avoidable_unseen_global = 0
    for fold in range(n_splits):
        train = frame.loc[folds != fold]
        valid = frame.loc[folds == fold]
        train_strata = set(train["stratum"])
        avoidable_unsupported += int(
            valid["stratum"].map(
                lambda value: value not in train_strata and stratum_totals[value] > 1
            ).sum()
        )
        train_ids = set(train["individual_id"])
        unseen_mask = ~valid["individual_id"].isin(train_ids)
        unseen_global += int(unseen_mask.sum())
        avoidable_unseen_global += int(
            valid["individual_id"].map(
                lambda value: value not in train_ids
                and id_source_group_totals[value] > 1
            ).sum()
        )

    components = {
        "avoidable_unsupported_rows": float(avoidable_unsupported),
        "unseen_global_id_rows": float(unseen_global),
        "avoidable_unseen_global_id_rows": float(avoidable_unseen_global),
        "stratum_variance": stratum_variance,
        "stratum_presence_loss": stratum_presence,
        "id_variance": id_variance,
        "id_presence_loss": id_presence,
        "part_balance_cv2": part_cv2,
        "size_balance_cv2": size_cv2,
    }
    score = (
        1000.0 * avoidable_unsupported
        + 100.0 * avoidable_unseen_global
        + 10.0 * stratum_presence
        + 3.0 * id_presence
        + 2.0 * stratum_variance
        + id_variance
        + part_cv2
        + size_cv2
    )
    return float(score), components


def audit_manifest(frame: pd.DataFrame, n_splits: int) -> dict[str, Any]:
    if frame["sample_index"].nunique() != len(frame):
        raise AssertionError("sample_index is not unique")
    if sorted(frame["fold"].unique().tolist()) != list(range(n_splits)):
        raise AssertionError("Fold IDs are incomplete")
    group_fold_counts = frame.groupby("source_group")["fold"].nunique()
    crossing = int((group_fold_counts > 1).sum())
    if crossing:
        raise AssertionError(f"{crossing} source groups cross folds")

    stratum_totals = frame["stratum"].value_counts()
    id_totals = frame["individual_id"].value_counts()
    id_source_group_totals = frame.groupby("individual_id")["source_group"].nunique()
    fold_reports: list[dict[str, Any]] = []
    for fold in range(n_splits):
        train = frame.loc[frame["fold"] != fold]
        valid = frame.loc[frame["fold"] == fold]
        train_strata = set(train["stratum"])
        train_ids = set(train["individual_id"])
        unsupported_part_mask = ~valid["stratum"].isin(train_strata)
        unseen_id_mask = ~valid["individual_id"].isin(train_ids)
        avoidable_mask = valid["stratum"].map(
            lambda value: value not in train_strata and stratum_totals[value] > 1
        )
        avoidable_global_mask = valid["individual_id"].map(
            lambda value: value not in train_ids and id_source_group_totals[value] > 1
        )
        fold_reports.append(
            {
                "fold": fold,
                "samples": int(len(valid)),
                "source_groups": int(valid["source_group"].nunique()),
                "unique_ids": int(valid["individual_id"].nunique()),
                "part_counts": {
                    part: int((valid["part"] == part).sum()) for part in PARTS
                },
                "part_unique_ids": {
                    part: int(valid.loc[valid["part"] == part, "individual_id"].nunique())
                    for part in PARTS
                },
                "part_id_cold_start_rows": int(unsupported_part_mask.sum()),
                "avoidable_part_id_cold_start_rows": int(avoidable_mask.sum()),
                "global_id_cold_start_rows": int(unseen_id_mask.sum()),
                "avoidable_global_id_cold_start_rows": int(avoidable_global_mask.sum()),
            }
        )

    stratum_fold_presence = frame.groupby("stratum")["fold"].nunique()
    return {
        "n_samples": int(len(frame)),
        "n_source_groups": int(frame["source_group"].nunique()),
        "n_individual_ids": int(frame["individual_id"].nunique()),
        "part_counts": {part: int((frame["part"] == part).sum()) for part in PARTS},
        "source_groups_crossing_folds": crossing,
        "singleton_part_id_strata": int((stratum_totals == 1).sum()),
        "single_row_global_ids": int((id_totals == 1).sum()),
        "single_source_global_ids": int((id_source_group_totals == 1).sum()),
        "part_id_fold_presence_histogram": {
            str(int(key)): int(value)
            for key, value in stratum_fold_presence.value_counts().sort_index().items()
        },
        "folds": fold_reports,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--candidates", type=int, default=256)
    args = parser.parse_args()
    if args.n_splits != 5:
        raise ValueError("This benchmark contract requires exactly five folds")
    if args.candidates < 1:
        raise ValueError("--candidates must be positive")

    frame = parse_official_lists(args.competition_root.resolve())
    best: tuple[float, int, np.ndarray, dict[str, float]] | None = None
    for offset in range(args.candidates):
        seed = args.seed + offset
        folds = make_candidate(frame, seed, args.n_splits)
        objective, components = candidate_objective(frame, folds, args.n_splits)
        candidate = (objective, seed, folds, components)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    objective, selected_seed, folds, components = best
    frame["fold"] = folds.astype(int)
    report = audit_manifest(frame, args.n_splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "folds.csv"
    frame.to_csv(manifest_path, index=False, encoding="utf-8")
    labels = (
        frame[["individual_id", "label_index"]]
        .drop_duplicates()
        .sort_values("label_index")
    )
    (args.output_dir / "label_map.json").write_text(
        json.dumps(
            {
                "labels": labels["individual_id"].tolist(),
                "part_to_index": {part: index for index, part in enumerate(PARTS)},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report.update(
        {
            "fold_builder": "multi-start StratifiedGroupKFold",
            "base_seed": args.seed,
            "candidate_count": args.candidates,
            "selected_seed": selected_seed,
            "objective": objective,
            "objective_components": components,
            "manifest_sha256": sha256_file(manifest_path),
        }
    )
    report_path = args.output_dir / "fold_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
