from __future__ import annotations

import argparse
import json
import math
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import torch

from data import (
    build_crop_geometry_stats,
    make_eval_loader,
    make_train_loader,
    validate_foreground_mask_artifact,
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
from train_full_v4_2 import (
    EXPECTED_TEMPLATE_SHA256,
    EXPECTED_TEST_MASK_INDEX_SHA256,
    EXPECTED_TRAIN_MASK_INDEX_SHA256,
    materialize_test_part_views,
    replay_sampler_exposure,
    set_model_metadata,
    validate_test_mask_artifact,
    validate_train_mask_provenance,
)


MODEL_VARIANT = (
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert"
)
PRETRAINING_REVISION = "dd0addc09788111fa893d24799a44744ba022eee"
AUGMENTATION_PROFILE = "arbase_source_jitter"
SAMPLER_PROFILE = "balanced_cross_part"
FOREGROUND_MODE = "sam_part_view"
TRAIN_EPOCHS = 32
SCHEDULER_EPOCHS = 48
WARMUP_EPOCHS = 3
IMAGE_SIZE = 448
EMBEDDING_DIM = 512
LOCAL_QUERIES = 4
IDENTITIES_PER_BATCH = 16
IMAGES_PER_IDENTITY = 4
FREEZE_BLOCKS = 24
FREEZE_STAGES = 2
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
CV_REFERENCE_NON_TTA = 0.7187289642816442
CV_REFERENCE_FLIP_DIAGNOSTIC = 0.7175879586792785
ZIP_NAME = "V6.1-full-noTTA.zip"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--train-foreground-mask-root", type=Path, required=True)
    parser.add_argument("--test-foreground-mask-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/v6_1_full_non_tta",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    source_dir = Path(__file__).resolve().parent
    paths = {
        "train_full_v6_1.py": Path(__file__).resolve(),
        "model.py": source_dir / "model.py",
        "data.py": source_dir / "data.py",
        "engine.py": source_dir / "engine.py",
        "losses.py": source_dir / "losses.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def main() -> None:
    args = parse_args()
    args.competition_root = args.competition_root.resolve()
    args.manifest = args.manifest.resolve()
    args.template = args.template.resolve()
    args.train_foreground_mask_root = args.train_foreground_mask_root.resolve()
    args.test_foreground_mask_root = args.test_foreground_mask_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.workers < 0 or args.eval_batch_size < 1:
        raise ValueError("Invalid loader resource setting")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validate_model_augmentation_profile(MODEL_VARIANT, AUGMENTATION_PROFILE)
    manifest, template, test_frame, labels = validate_inputs(args)
    if sha256_file(args.template) != EXPECTED_TEMPLATE_SHA256:
        raise AssertionError("Official template SHA-256 changed")
    if manifest.image_path.astype(str).str.startswith("test/").any():
        raise AssertionError("Anonymous test image entered full training")
    train_names = set(
        manifest.image_path.astype(str).map(Path).map(lambda path: path.name)
    )
    if train_names.intersection(template.image_id.astype(str)):
        raise AssertionError("Train/test filename overlap")

    geometry_stats = build_crop_geometry_stats(manifest, args.competition_root)
    if geometry_stats["rows"] != 4067:
        raise AssertionError("Full-data geometry stats changed")
    _, train_foreground_metadata = validate_foreground_mask_artifact(
        manifest,
        args.train_foreground_mask_root,
        verify_mask_hashes=True,
    )
    train_foreground_summary = validate_train_mask_provenance(
        args.train_foreground_mask_root
    )
    test_mask_index, test_foreground_metadata = validate_test_mask_artifact(
        test_frame, args.test_foreground_mask_root
    )

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
        geometry_stats=geometry_stats,
        foreground_mask_root=args.train_foreground_mask_root,
        foreground_mode=FOREGROUND_MODE,
    )
    if len(train_loader) != 64:
        raise AssertionError(f"Expected 64 full-data steps, got {len(train_loader)}")
    sampler_exposure = replay_sampler_exposure(train_sampler, manifest)

    device = torch.device("cuda")
    model = build_model(
        MODEL_VARIANT,
        num_classes=len(labels),
        image_size=IMAGE_SIZE,
        embedding_dim=EMBEDDING_DIM,
        local_queries=LOCAL_QUERIES,
        pretrained=True,
        arc_scale=ARC_SCALE,
        arc_margin=ARC_MARGIN,
        part_delta_scale=PART_DELTA_SCALE,
        freeze_blocks=FREEZE_BLOCKS,
        freeze_stages=FREEZE_STAGES,
        grad_checkpointing=True,
    ).to(device)
    set_model_metadata(model, manifest, len(labels))
    if (
        not model.head_identity_expert
        or model.head_tail_other_expert
        or not model.head_identity_expert_routing_enabled
        or abs(model.head_identity_expert_mix - 0.25) > 1e-12
        or abs(model.head_identity_expert_loss_weight - 0.50) > 1e-12
    ):
        raise AssertionError("Formal V6.1 head route changed")
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
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)

    config = {
        "competition_root": str(args.competition_root),
        "manifest": str(args.manifest),
        "template": str(args.template),
        "train_foreground_mask_root": str(args.train_foreground_mask_root),
        "test_foreground_mask_root": str(args.test_foreground_mask_root),
        "output_dir": str(args.output_dir),
        "model_variant": MODEL_VARIANT,
        "model_name": model.model_name,
        "pretraining_source": model.pretraining_source,
        "pretraining_revision": PRETRAINING_REVISION,
        "init_checkpoint": None,
        "competition_checkpoint_loaded": False,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "sampler_profile": SAMPLER_PROFILE,
        "sampler_exposure": sampler_exposure,
        "foreground_mode": FOREGROUND_MODE,
        "foreground_aux_weight": 0.0,
        "train_foreground_mask_artifact": train_foreground_metadata,
        "train_foreground_summary": train_foreground_summary,
        "test_foreground_mask_artifact": test_foreground_metadata,
        "released_crop_geometry_stats": geometry_stats,
        "train_epochs": TRAIN_EPOCHS,
        "scheduler_horizon_epochs": SCHEDULER_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "image_size": IMAGE_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "local_queries": LOCAL_QUERIES,
        "identities_per_batch": IDENTITIES_PER_BATCH,
        "images_per_identity": IMAGES_PER_IDENTITY,
        "freeze_blocks": FREEZE_BLOCKS,
        "freeze_stages": FREEZE_STAGES,
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
        "amp_initial_scale": 1024.0,
        "seed": SEED,
        "tta_flip": False,
        "gallery_decoder": False,
        "head_expert_route": {
            "enabled": True,
            "scope": "head only",
            "base_weight": 0.75,
            "expert_weight": 0.25,
            "availability_fallback": True,
        },
        "full_epoch_rule": (
            "V6.1 fold-0 routed non-TTA best human epoch 32 fixed before test inference"
        ),
        "cv_reference_non_tta": CV_REFERENCE_NON_TTA,
        "cv_reference_flip_diagnostic": CV_REFERENCE_FLIP_DIAGNOSTIC,
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
            foreground_aux_weight=0.0,
        )
        record = {
            "epoch": epoch,
            **losses,
            "learning_rate_max": max(group["lr"] for group in optimizer.param_groups),
            "seconds": time.time() - epoch_started,
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        if not math.isfinite(float(record["loss"])) or record["gpu_peak_gb"] >= 24.0:
            raise FloatingPointError("Full-data epoch failed numerical or memory gate")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"full epoch={epoch + 1:02d}/{TRAIN_EPOCHS} "
            f"loss={losses['loss']:.4f} part_ce={losses['part_ce']:.4f} "
            f"shared_ce={losses['shared_ce']:.4f} "
            f"head_expert_ce={losses['head_expert_ce']:.4f} "
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
    records = [
        line for line in log_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(records) != TRAIN_EPOCHS:
        raise AssertionError("Full-data training log is incomplete")

    inference_frame, test_view_metadata = materialize_test_part_views(
        test_frame,
        args.competition_root,
        args.test_foreground_mask_root,
        test_mask_index,
        args.output_dir / "test_body_views",
    )
    test_loader = make_eval_loader(
        inference_frame,
        args.competition_root,
        IMAGE_SIZE,
        args.eval_batch_size,
        args.workers,
        SEED,
        augmentation_profile=AUGMENTATION_PROFILE,
        geometry_stats=geometry_stats,
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
    predictions_path = args.output_dir / "test_predictions.csv"
    predicted.to_csv(predictions_path, index=False, encoding="utf-8")
    prediction_lookup = predicted.set_index("image_id").predicted_id
    submission = template[["image_id"]].copy()
    submission["predicted_id"] = submission.image_id.map(prediction_lookup)
    if (
        submission.predicted_id.isna().any()
        or not set(submission.predicted_id).issubset(set(labels))
        or submission.image_id.astype(str).tolist()
        != template.image_id.astype(str).tolist()
        or len(submission) != 1641
        or submission.image_id.nunique() != 1641
    ):
        raise AssertionError("Submission coverage or label space changed")

    csv_path = args.output_dir / "submission.csv"
    submission.to_csv(csv_path, index=False, encoding="utf-8")
    zip_path = args.output_dir / ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["submission.csv"]:
            raise AssertionError(f"Unexpected ZIP members: {archive.namelist()}")
        archived = pd.read_csv(archive.open("submission.csv"))
    if not archived.equals(submission):
        raise AssertionError("Archived CSV differs from validated predictions")

    report = {
        "elapsed_seconds": time.time() - started,
        "train_epochs": TRAIN_EPOCHS,
        "scheduler_horizon_epochs": SCHEDULER_EPOCHS,
        "train_rows": len(manifest),
        "test_rows": len(submission),
        "tta_flip": False,
        "decoder": "classifier",
        "head_expert_route": config["head_expert_route"],
        "test_observation": "sam_part_view",
        "test_view_artifact": test_view_metadata,
        "unique_predicted_ids": int(submission.predicted_id.nunique()),
        "unique_predicted_ids_per_part": (
            predicted.groupby("part").predicted_id.nunique().astype(int).to_dict()
        ),
        "manifest_sha256": config["manifest_sha256"],
        "template_sha256": config["template_sha256"],
        "train_foreground_index_sha256": EXPECTED_TRAIN_MASK_INDEX_SHA256,
        "test_foreground_index_sha256": EXPECTED_TEST_MASK_INDEX_SHA256,
        "source_sha256": config["source_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test_predictions_sha256": sha256_file(predictions_path),
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
