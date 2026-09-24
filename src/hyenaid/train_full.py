from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from data import make_eval_loader, make_train_loader
from engine import (
    build_optimizer,
    build_scheduler,
    extract,
    prediction_frame,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    write_json,
)
from model import build_model


EXPECTED_MANIFEST_SHA256 = (
    "0470a4bb4630061ef06e92c50a74cb15ba457e476714294872ec426239a1ee6f"
)
EXPECTED_PART_ROWS = {"head": 798, "left_body": 414, "right_body": 429}
PART_INFO = {
    "head_": ("head", 0, "headTest"),
    "left_": ("left_body", 1, "leftTest"),
    "right_": ("right_body", 2, "rightTest"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def make_test_frame(template: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_index, image_id in enumerate(template.image_id.astype(str)):
        matches = [info for prefix, info in PART_INFO.items() if image_id.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous test image prefix: {image_id}")
        part, part_index, directory = matches[0]
        rows.append(
            {
                "sample_index": sample_index,
                "image_id": image_id,
                "image_path": f"test/{directory}/{image_id}",
                "source_group": image_id,
                "part": part,
                "part_index": part_index,
                "individual_id": "",
                "label_index": -1,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs/v2_9_full"
    )
    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--images-per-identity", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--backbone-lr", type=float, default=2.4e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--layer-decay", type=float, default=0.82)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--arc-scale", type=float, default=30.0)
    parser.add_argument("--arc-margin", type=float, default=0.20)
    parser.add_argument("--part-delta-scale", type=float, default=0.20)
    parser.add_argument("--shared-ce-weight", type=float, default=0.50)
    parser.add_argument("--supcon-weight", type=float, default=0.15)
    parser.add_argument("--triplet-weight", type=float, default=0.30)
    parser.add_argument("--supcon-temperature", type=float, default=0.10)
    parser.add_argument("--triplet-scale", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise AssertionError(f"Unexpected manifest hash: {manifest_hash}")
    manifest = pd.read_csv(args.manifest)
    if len(manifest) != 4067 or manifest.fold.nunique() != 5:
        raise AssertionError("Unexpected full training manifest")
    if manifest.image_path.str.startswith("test/").any():
        raise AssertionError("Anonymous test image found in training manifest")
    if manifest.sample_index.nunique() != len(manifest):
        raise AssertionError("Duplicate training sample_index")
    labels = (
        manifest[["individual_id", "label_index"]]
        .drop_duplicates()
        .sort_values("label_index")
        .individual_id.astype(str)
        .tolist()
    )
    if len(labels) != 255:
        raise AssertionError(f"Expected 255 labels, got {len(labels)}")
    expected_indices = list(range(len(labels)))
    actual_indices = sorted(manifest.label_index.unique().tolist())
    if actual_indices != expected_indices:
        raise AssertionError("Training label indices are not contiguous")

    template = pd.read_csv(args.template)
    if list(template.columns) != ["image_id", "predicted_id"]:
        raise AssertionError(f"Unexpected template columns: {list(template.columns)}")
    if len(template) != 1641 or template.image_id.nunique() != 1641:
        raise AssertionError("Unexpected submission template rows")
    test_frame = make_test_frame(template)
    part_rows = test_frame.part.value_counts().to_dict()
    if part_rows != EXPECTED_PART_ROWS:
        raise AssertionError(f"Unexpected test part counts: {part_rows}")
    missing = [
        path
        for path in test_frame.image_path
        if not (args.competition_root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing anonymous test image: {missing[0]}")
    return manifest, template, test_frame, labels


def main() -> None:
    args = parse_args()
    args.competition_root = args.competition_root.resolve()
    args.manifest = args.manifest.resolve()
    args.template = args.template.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.epochs != 13:
        raise AssertionError("V2.9 full-data calibration has a locked 13-epoch rule")
    if (args.output_dir / "model.pt").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir / 'model.pt'}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    manifest, template, test_frame, labels = validate_inputs(args)

    train_loader, train_sampler = make_train_loader(
        manifest,
        args.competition_root,
        args.image_size,
        args.workers,
        args.identities_per_batch,
        args.images_per_identity,
        args.seed,
        augmentation_profile="arbase",
        sampler_profile="cross_part",
    )
    device = torch.device("cuda")
    model = build_model(
        "dinov3_large_patch_mgn",
        num_classes=len(labels),
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        local_queries=4,
        pretrained=True,
        arc_scale=args.arc_scale,
        arc_margin=args.arc_margin,
        part_delta_scale=args.part_delta_scale,
        freeze_blocks=0,
        freeze_stages=2,
        grad_checkpointing=True,
    ).to(device)
    availability = torch.zeros(3, len(labels), dtype=torch.bool)
    availability[
        torch.as_tensor(manifest.part_index.to_numpy(copy=True)),
        torch.as_tensor(manifest.label_index.to_numpy(copy=True)),
    ] = True
    model.set_part_availability(availability)
    if hasattr(model, "set_part_counts"):
        part_counts = torch.zeros(3, len(labels), dtype=torch.float32)
        grouped_counts = manifest.groupby(["part_index", "label_index"]).size()
        for (part_index, label_index), count in grouped_counts.items():
            part_counts[int(part_index), int(label_index)] = float(count)
        model.set_part_counts(part_counts)
    if hasattr(model, "set_class_counts"):
        class_counts = torch.bincount(
            torch.as_tensor(manifest.label_index.to_numpy(copy=True)),
            minlength=len(labels),
        )
        model.set_class_counts(class_counts)

    optimizer = build_optimizer(
        model,
        args.backbone_lr,
        args.head_lr,
        args.weight_decay,
        args.layer_decay,
        optimizer_name="adamw",
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=args.epochs * len(train_loader),
        warmup_steps=args.warmup_epochs * len(train_loader),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    config = serializable_args(args)
    config.update(
        {
            "model_variant": "dinov3_large_patch_mgn",
            "model_name": model.model_name,
            "pretraining_source": model.pretraining_source,
            "augmentation_profile": "arbase",
            "sampler_profile": "cross_part",
            "optimizer": "adamw",
            "freeze_blocks": 0,
            "freeze_stages": 2,
            "tta_flip": True,
            "gallery_decoder": False,
            "epoch_rule": "V2.9 fold-0 non-TTA best epoch fixed before test inference",
            "manifest_sha256": sha256_file(args.manifest),
            "template_sha256": sha256_file(args.template),
            "train_samples": len(manifest),
            "test_samples": len(test_frame),
            "train_steps_per_epoch": len(train_loader),
            "torch_version": torch.__version__,
        }
    )
    write_json(config, args.output_dir / "config.json")
    log_path = args.output_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    started = time.time()
    for epoch in range(args.epochs):
        epoch_started = time.time()
        torch.cuda.reset_peak_memory_stats()
        train_sampler.set_epoch(epoch)
        losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            label_smoothing=args.label_smoothing,
            shared_ce_weight=args.shared_ce_weight,
            supcon_weight=args.supcon_weight,
            triplet_weight=args.triplet_weight,
            prototype_weight=0.0,
            supcon_temperature=args.supcon_temperature,
            triplet_scale=args.triplet_scale,
            grad_clip=args.grad_clip,
        )
        record = {
            "epoch": epoch,
            **losses,
            "learning_rate_max": max(group["lr"] for group in optimizer.param_groups),
            "seconds": time.time() - epoch_started,
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"full epoch={epoch + 1:02d}/{args.epochs} "
            f"loss={losses['loss']:.4f} part_ce={losses['part_ce']:.4f} "
            f"shared_ce={losses['shared_ce']:.4f} "
            f"proto_ce={losses['prototype_ce']:.4f} "
            f"time={record['seconds']:.1f}s mem={record['gpu_peak_gb']:.1f}GB",
            flush=True,
        )

    checkpoint_path = args.output_dir / "model.pt"
    save_checkpoint(
        {
            "model": model.state_dict(),
            "epoch": args.epochs - 1,
            "labels": labels,
            "config": config,
        },
        checkpoint_path,
    )

    # Anonymous test pixels are loaded only after the fixed final checkpoint
    # has been written. Inference uses V2.9's locked classifier and flip TTA.
    test_loader = make_eval_loader(
        test_frame,
        args.competition_root,
        args.image_size,
        args.eval_batch_size,
        args.workers,
        args.seed,
        augmentation_profile="arbase",
    )
    test_features = extract(
        model,
        test_loader,
        device,
        tta_flip=True,
        return_local=False,
        local_grid=6,
    )
    predicted = prediction_frame(
        test_frame,
        test_features,
        test_features["classifier_score"],
        labels,
    )
    prediction_lookup = predicted.set_index("image_id").predicted_id
    submission = template[["image_id"]].copy()
    submission["predicted_id"] = submission.image_id.map(prediction_lookup)
    if submission.predicted_id.isna().any():
        raise AssertionError("Missing test prediction")
    if not set(submission.predicted_id).issubset(set(labels)):
        raise AssertionError("Prediction outside the official training label space")
    if submission.image_id.tolist() != template.image_id.astype(str).tolist():
        raise AssertionError("Submission row order differs from the official template")
    if len(submission) != 1641 or submission.image_id.nunique() != 1641:
        raise AssertionError("Invalid submission row count or duplicate image_id")

    csv_path = args.output_dir / "submission.csv"
    submission.to_csv(csv_path, index=False, encoding="utf-8")
    zip_path = args.output_dir / "submission_v2_9.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["submission.csv"]:
            raise AssertionError(f"Unexpected ZIP members: {archive.namelist()}")
        archived = pd.read_csv(archive.open("submission.csv"))
    if not archived.equals(submission):
        raise AssertionError("Archived CSV differs from the validated submission")

    per_part_unique = (
        predicted.groupby("part").predicted_id.nunique().astype(int).to_dict()
    )
    report = {
        "elapsed_seconds": time.time() - started,
        "epochs": args.epochs,
        "train_rows": len(manifest),
        "test_rows": len(submission),
        "unique_predicted_ids": int(submission.predicted_id.nunique()),
        "unique_predicted_ids_per_part": per_part_unique,
        "manifest_sha256": config["manifest_sha256"],
        "template_sha256": config["template_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "submission_csv_sha256": sha256_file(csv_path),
        "submission_zip_sha256": sha256_file(zip_path),
        "checkpoint": str(checkpoint_path),
        "submission_csv": str(csv_path),
        "submission_zip": str(zip_path),
    }
    write_json(report, args.output_dir / "inference_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    main()
