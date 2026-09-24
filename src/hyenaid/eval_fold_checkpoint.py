from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from data import (
    build_crop_geometry_stats,
    make_eval_loader,
    validate_foreground_mask_artifact,
)
from engine import competition_metrics, extract, prediction_frame, write_json
from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-foreground-mask-root", type=Path)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    train = manifest.loc[manifest.fold != args.fold].copy()
    valid = manifest.loc[manifest.fold == args.fold].copy()
    if set(train.source_group).intersection(valid.source_group):
        raise AssertionError("Source leakage")
    labels = (
        manifest[["individual_id", "label_index"]]
        .drop_duplicates()
        .sort_values("label_index")
        .individual_id.astype(str)
        .tolist()
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    if int(config["fold"]) != args.fold:
        raise AssertionError("Checkpoint fold mismatch")
    model = build_model(
        config["model_variant"],
        num_classes=len(labels),
        image_size=int(config["image_size"]),
        embedding_dim=int(config["embedding_dim"]),
        local_queries=int(config["local_queries"]),
        pretrained=False,
        arc_scale=float(config["arc_scale"]),
        arc_margin=float(config["arc_margin"]),
        part_delta_scale=float(config["part_delta_scale"]),
        freeze_blocks=int(config["freeze_blocks"]),
        freeze_stages=int(config["freeze_stages"]),
        grad_checkpointing=True,
    )
    availability = torch.zeros(3, len(labels), dtype=torch.bool)
    availability[
        torch.as_tensor(train.part_index.to_numpy(copy=True)),
        torch.as_tensor(train.label_index.to_numpy(copy=True)),
    ] = True
    model.set_part_availability(availability)
    if hasattr(model, "set_part_counts"):
        counts = torch.zeros(3, len(labels), dtype=torch.float32)
        for (part, label), count in train.groupby(
            ["part_index", "label_index"]
        ).size().items():
            counts[int(part), int(label)] = float(count)
        model.set_part_counts(counts)
    if hasattr(model, "set_class_counts"):
        model.set_class_counts(
            torch.bincount(
                torch.as_tensor(train.label_index.to_numpy(copy=True)),
                minlength=len(labels),
            )
        )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda")
    model.to(device)
    geometry_stats = config.get("released_crop_geometry_stats")
    if getattr(model, "continuous_geometry_conditioning", False):
        if geometry_stats is None:
            raise AssertionError("Geometry checkpoint omitted fold-train statistics")
        recomputed_geometry_stats = build_crop_geometry_stats(
            train, args.competition_root
        )
        if recomputed_geometry_stats != geometry_stats:
            raise AssertionError("Frozen geometry statistics are not reproducible")
    elif geometry_stats is not None:
        raise AssertionError("Non-geometry model carries geometry statistics")
    foreground_mode = str(config.get("foreground_mode", "none"))
    validation_foreground_root: Path | None = None
    if foreground_mode in {"sam_view", "sam_part_view"}:
        configured_root = config.get("validation_foreground_mask_root")
        if args.validation_foreground_mask_root is not None:
            validation_foreground_root = (
                args.validation_foreground_mask_root.resolve()
            )
        elif configured_root is not None:
            validation_foreground_root = Path(configured_root).resolve()
        else:
            raise AssertionError("SAM-view checkpoint omitted validation masks")
        _, validation_foreground_metadata = validate_foreground_mask_artifact(
            valid,
            validation_foreground_root,
            verify_mask_hashes=True,
        )
        stored_validation_foreground = config.get(
            "validation_foreground_mask_artifact"
        )
        if not isinstance(stored_validation_foreground, dict) or any(
            validation_foreground_metadata.get(key)
            != stored_validation_foreground.get(key)
            for key in (
                "index_sha256",
                "rows",
                "valid",
                "valid_fraction",
                "by_part",
                "mask_hashes_verified",
            )
        ):
            raise AssertionError("Validation foreground provenance changed")
    elif (
        args.validation_foreground_mask_root is not None
        or config.get("validation_foreground_mask_artifact") is not None
    ):
        raise AssertionError("Validation masks require the fixed SAM-view mode")
    loader = make_eval_loader(
        valid,
        args.competition_root,
        int(config["image_size"]),
        int(config["eval_batch_size"]),
        int(config["workers"]),
        int(config["fold_seed"]),
        augmentation_profile=config["augmentation_profile"],
        geometry_stats=geometry_stats,
        foreground_mask_root=validation_foreground_root,
        foreground_mode=(
            foreground_mode
            if foreground_mode in {"sam_view", "sam_part_view"}
            else "none"
        ),
    )
    features = extract(
        model,
        loader,
        device,
        tta_flip=True,
        return_local=False,
        local_grid=int(config["local_grid"]),
    )
    frame = prediction_frame(
        manifest, features, features["classifier_score"], labels
    )
    if len(frame) != len(valid):
        raise AssertionError("Validation prediction coverage mismatch")
    metrics = competition_metrics(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.output_dir / "val_predictions.csv", index=False, encoding="utf-8"
    )
    log_path = args.output_dir / "train_log.jsonl"
    epochs_ran = sum(1 for _ in log_path.open()) if log_path.is_file() else None
    result = {
        "fold": args.fold,
        "best_epoch": int(checkpoint["epoch"]),
        "epochs_ran": epochs_ran,
        "selection_classifier_metrics": checkpoint["classifier_metrics"],
        "decoder": "classifier",
        "classifier_metrics": metrics,
        "final_metrics": metrics,
        "fused_metrics": metrics,
        "evaluation_only_after_manual_stop": True,
    }
    write_json(result, args.output_dir / "metrics.json")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
