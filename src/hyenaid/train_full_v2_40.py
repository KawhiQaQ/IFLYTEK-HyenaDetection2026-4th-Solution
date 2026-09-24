from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import torch

from data import (
    make_eval_loader,
    make_train_loader,
    validate_model_augmentation_profile,
)
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
from train_full import sha256_file, validate_inputs


MODEL_VARIANT = "dinov3_huge_plus_patch_mgn_cov"
PRETRAINING_REVISION = "dd0addc09788111fa893d24799a44744ba022eee"
AUGMENTATION_PROFILE = "arbase_source_jitter"
SAMPLER_PROFILE = "cross_part"
TRAIN_EPOCHS = 21
SCHEDULER_EPOCHS = 48
WARMUP_EPOCHS = 3
IMAGE_SIZE = 448
EMBEDDING_DIM = 512
IDENTITIES_PER_BATCH = 16
IMAGES_PER_IDENTITY = 4
FREEZE_BLOCKS = 24
BACKBONE_LR = 2.4e-5
HEAD_LR = 3.0e-4
LAYER_DECAY = 0.82
WEIGHT_DECAY = 0.05
ARC_SCALE = 30.0
ARC_MARGIN = 0.20
PART_DELTA_SCALE = 0.20
LABEL_SMOOTHING = 0.05
SHARED_CE_WEIGHT = 0.50
SUPCON_WEIGHT = 0.15
TRIPLET_WEIGHT = 0.30
SUPCON_TEMPERATURE = 0.10
TRIPLET_SCALE = 0.10
GRAD_CLIP = 1.0
SEED = 20260719


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/v2_40_full_non_tta",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    source_dir = Path(__file__).resolve().parent
    paths = {
        "train_full_v2_40.py": Path(__file__).resolve(),
        "model.py": source_dir / "model.py",
        "data.py": source_dir / "data.py",
        "engine.py": source_dir / "engine.py",
        "losses.py": source_dir / "losses.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def set_model_metadata(
    model: torch.nn.Module,
    manifest: pd.DataFrame,
    label_count: int,
) -> None:
    availability = torch.zeros(3, label_count, dtype=torch.bool)
    availability[
        torch.as_tensor(manifest.part_index.to_numpy(copy=True)),
        torch.as_tensor(manifest.label_index.to_numpy(copy=True)),
    ] = True
    model.set_part_availability(availability)

    if hasattr(model, "set_part_counts"):
        part_counts = torch.zeros(3, label_count, dtype=torch.float32)
        grouped = manifest.groupby(["part_index", "label_index"]).size()
        for (part_index, label_index), count in grouped.items():
            part_counts[int(part_index), int(label_index)] = float(count)
        model.set_part_counts(part_counts)
    if hasattr(model, "set_class_counts"):
        class_counts = torch.bincount(
            torch.as_tensor(manifest.label_index.to_numpy(copy=True)),
            minlength=label_count,
        )
        model.set_class_counts(class_counts)


def main() -> None:
    args = parse_args()
    args.competition_root = args.competition_root.resolve()
    args.manifest = args.manifest.resolve()
    args.template = args.template.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.workers < 0 or args.eval_batch_size < 1:
        raise ValueError("Invalid loader resource setting")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validate_model_augmentation_profile(MODEL_VARIANT, AUGMENTATION_PROFILE)
    manifest, template, test_frame, labels = validate_inputs(args)
    # Test membership is filename-based and must remain disjoint from every
    # public training path. Anonymous pixels are not decoded in this check.
    train_names = set(manifest.image_path.astype(str).map(Path).map(lambda p: p.name))
    overlap = train_names.intersection(template.image_id.astype(str))
    if overlap:
        raise AssertionError(f"Train/test filename overlap: {len(overlap)}")

    set_seed(SEED)
    train_loader, train_sampler = make_train_loader(
        manifest,
        args.competition_root,
        IMAGE_SIZE,
        args.workers,
        IDENTITIES_PER_BATCH,
        IMAGES_PER_IDENTITY,
        SEED,
        augmentation_profile=AUGMENTATION_PROFILE,
        sampler_profile=SAMPLER_PROFILE,
    )
    if len(train_loader) != 64:
        raise AssertionError(f"Expected 64 full-data steps, got {len(train_loader)}")

    device = torch.device("cuda")
    model = build_model(
        MODEL_VARIANT,
        num_classes=len(labels),
        image_size=IMAGE_SIZE,
        embedding_dim=EMBEDDING_DIM,
        local_queries=4,
        pretrained=True,
        arc_scale=ARC_SCALE,
        arc_margin=ARC_MARGIN,
        part_delta_scale=PART_DELTA_SCALE,
        freeze_blocks=FREEZE_BLOCKS,
        freeze_stages=2,
        grad_checkpointing=True,
    ).to(device)
    set_model_metadata(model, manifest, len(labels))
    optimizer = build_optimizer(
        model,
        BACKBONE_LR,
        HEAD_LR,
        WEIGHT_DECAY,
        LAYER_DECAY,
        optimizer_name="adamw",
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=SCHEDULER_EPOCHS * len(train_loader),
        warmup_steps=WARMUP_EPOCHS * len(train_loader),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    config = {
        "competition_root": str(args.competition_root),
        "manifest": str(args.manifest),
        "template": str(args.template),
        "output_dir": str(args.output_dir),
        "model_variant": MODEL_VARIANT,
        "model_name": model.model_name,
        "pretraining_source": model.pretraining_source,
        "pretraining_revision": PRETRAINING_REVISION,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "sampler_profile": SAMPLER_PROFILE,
        "train_epochs": TRAIN_EPOCHS,
        "scheduler_horizon_epochs": SCHEDULER_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "image_size": IMAGE_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "identities_per_batch": IDENTITIES_PER_BATCH,
        "images_per_identity": IMAGES_PER_IDENTITY,
        "freeze_blocks": FREEZE_BLOCKS,
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
        "layer_decay": LAYER_DECAY,
        "weight_decay": WEIGHT_DECAY,
        "arc_scale": ARC_SCALE,
        "arc_margin": ARC_MARGIN,
        "part_delta_scale": PART_DELTA_SCALE,
        "label_smoothing": LABEL_SMOOTHING,
        "shared_ce_weight": SHARED_CE_WEIGHT,
        "supcon_weight": SUPCON_WEIGHT,
        "triplet_weight": TRIPLET_WEIGHT,
        "prototype_weight": 0.0,
        "supcon_temperature": SUPCON_TEMPERATURE,
        "triplet_scale": TRIPLET_SCALE,
        "grad_clip": GRAD_CLIP,
        "optimizer": "adamw",
        "seed": SEED,
        "tta_flip": False,
        "gallery_decoder": False,
        "full_epoch_rule": "V2.40 fold-0 non-TTA best epoch fixed before test inference",
        "cv_reference_non_tta": 0.6527427923419687,
        "cv_reference_flip_tta": 0.6478412560578092,
        "manifest_sha256": sha256_file(args.manifest),
        "template_sha256": sha256_file(args.template),
        "source_sha256": source_hashes(),
        "train_samples": len(manifest),
        "test_samples": len(test_frame),
        "train_steps_per_epoch": len(train_loader),
        "workers": args.workers,
        "eval_batch_size": args.eval_batch_size,
        "torch_version": torch.__version__,
    }
    write_json(config, args.output_dir / "config.json")
    log_path = args.output_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    started = time.time()
    for epoch in range(TRAIN_EPOCHS):
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
            label_smoothing=LABEL_SMOOTHING,
            shared_ce_weight=SHARED_CE_WEIGHT,
            supcon_weight=SUPCON_WEIGHT,
            triplet_weight=TRIPLET_WEIGHT,
            prototype_weight=0.0,
            supcon_temperature=SUPCON_TEMPERATURE,
            triplet_scale=TRIPLET_SCALE,
            grad_clip=GRAD_CLIP,
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
            f"full epoch={epoch + 1:02d}/{TRAIN_EPOCHS} "
            f"loss={losses['loss']:.4f} part_ce={losses['part_ce']:.4f} "
            f"shared_ce={losses['shared_ce']:.4f} "
            f"time={record['seconds']:.1f}s mem={record['gpu_peak_gb']:.1f}GB",
            flush=True,
        )

    checkpoint_path = args.output_dir / "model.pt"
    save_checkpoint(
        {
            "model": model.state_dict(),
            "epoch": TRAIN_EPOCHS - 1,
            "labels": labels,
            "config": config,
        },
        checkpoint_path,
    )

    # Anonymous pixels are first decoded here, after the final fixed checkpoint
    # exists. V2.40's selected protocol is a single deterministic non-TTA pass.
    test_loader = make_eval_loader(
        test_frame,
        args.competition_root,
        IMAGE_SIZE,
        args.eval_batch_size,
        args.workers,
        SEED,
        augmentation_profile=AUGMENTATION_PROFILE,
    )
    test_features = extract(
        model,
        test_loader,
        device,
        tta_flip=False,
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
        raise AssertionError("Missing anonymous-test prediction")
    if not set(submission.predicted_id).issubset(set(labels)):
        raise AssertionError("Prediction outside the official 255-label space")
    if submission.image_id.astype(str).tolist() != template.image_id.astype(str).tolist():
        raise AssertionError("Submission order differs from the official template")
    if len(submission) != 1641 or submission.image_id.nunique() != 1641:
        raise AssertionError("Invalid submission coverage")

    csv_path = args.output_dir / "submission.csv"
    submission.to_csv(csv_path, index=False, encoding="utf-8")
    zip_path = args.output_dir / "V2.40-noTTA.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["submission.csv"]:
            raise AssertionError(f"Unexpected ZIP members: {archive.namelist()}")
        archived = pd.read_csv(archive.open("submission.csv"))
    if not archived.equals(submission):
        raise AssertionError("Archived CSV differs from validated predictions")

    per_part_unique = (
        predicted.groupby("part").predicted_id.nunique().astype(int).to_dict()
    )
    report = {
        "elapsed_seconds": time.time() - started,
        "train_epochs": TRAIN_EPOCHS,
        "scheduler_horizon_epochs": SCHEDULER_EPOCHS,
        "train_rows": len(manifest),
        "test_rows": len(submission),
        "tta_flip": False,
        "unique_predicted_ids": int(submission.predicted_id.nunique()),
        "unique_predicted_ids_per_part": per_part_unique,
        "manifest_sha256": config["manifest_sha256"],
        "template_sha256": config["template_sha256"],
        "source_sha256": config["source_sha256"],
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
