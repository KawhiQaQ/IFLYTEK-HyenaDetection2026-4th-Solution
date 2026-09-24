from __future__ import annotations

import argparse
import json
import math
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from data import (
    build_crop_geometry_stats,
    foreground_safe_background_blur,
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


MODEL_VARIANT = "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2"
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
CV_REFERENCE_NON_TTA = 0.7074277886577711
CV_REFERENCE_FLIP_DIAGNOSTIC = 0.706745291189227
EXPECTED_TRAIN_MASK_INDEX_SHA256 = (
    "53fbb173a1ace9a5cdbf8054f7c7b5747e9e30d639940ad13d4f2e5304be8687"
)
EXPECTED_TEST_MASK_INDEX_SHA256 = (
    "7aba1a44ec180df5b6b6257807ace7a81b693e3212efdef1d7707b7f005a2750"
)
EXPECTED_TEMPLATE_SHA256 = (
    "274c7db5bbc34763e1081dcd21b88abf0e087b761d13beb8b35fdc464d76f2f8"
)
EXPECTED_SAM2_CHECKPOINT_SHA256 = (
    "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
)
EXPECTED_SAM2_CONFIG_SHA256 = (
    "f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d"
)
EXPECTED_SAM2_SOURCE_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
EXPECTED_TEST_MASK_GENERATOR_SHA256 = (
    "3ba258195aac455b4f05e1e01d5cb1dde01ecff26d095b5a3b333cedbd736cee"
)
EXPECTED_TEST_VALID = 1399
EXPECTED_TEST_BODY_VIEWS = 758


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
        default=root / "outputs/v4_2_full_non_tta",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    source_dir = Path(__file__).resolve().parent
    paths = {
        "train_full_v4_2.py": Path(__file__).resolve(),
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
        for (part_index, label_index), count in (
            manifest.groupby(["part_index", "label_index"]).size().items()
        ):
            part_counts[int(part_index), int(label_index)] = float(count)
        model.set_part_counts(part_counts)
    if hasattr(model, "set_class_counts"):
        model.set_class_counts(
            torch.bincount(
                torch.as_tensor(manifest.label_index.to_numpy(copy=True)),
                minlength=label_count,
            )
        )


def validate_train_mask_provenance(mask_root: Path) -> dict[str, Any]:
    summary_path = mask_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "scope": "all_train",
        "fold": None,
        "rows": 4067,
        "valid": 3407,
        "index_sha256": EXPECTED_TRAIN_MASK_INDEX_SHA256,
        "sam2_checkpoint_sha256": EXPECTED_SAM2_CHECKPOINT_SHA256,
        "sam2_config_sha256": EXPECTED_SAM2_CONFIG_SHA256,
        "sam2_source_commit": EXPECTED_SAM2_SOURCE_COMMIT,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise AssertionError(
                f"Full-train foreground summary {key}={summary.get(key)!r}, "
                f"expected {value!r}"
            )
    if sha256_file(mask_root / "index.csv") != EXPECTED_TRAIN_MASK_INDEX_SHA256:
        raise AssertionError("Full-train foreground index SHA-256 changed")
    return summary


def validate_test_mask_artifact(
    test_frame: pd.DataFrame,
    mask_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    index_path = mask_root / "index.csv"
    summary_path = mask_root / "summary.json"
    if not index_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(index_path if not index_path.is_file() else summary_path)
    index = pd.read_csv(index_path)
    required_columns = {
        "sample_index",
        "image_path",
        "source_group",
        "part",
        "part_index",
        "valid",
        "method",
        "candidate_index",
        "score",
        "area",
        "mask_sha256",
    }
    if set(index.columns) != required_columns:
        raise AssertionError("Unexpected anonymous-test foreground columns")
    expected = test_frame[
        ["sample_index", "image_path", "source_group", "part", "part_index"]
    ].sort_values("sample_index").reset_index(drop=True)
    actual = index[
        ["sample_index", "image_path", "source_group", "part", "part_index"]
    ].sort_values("sample_index").reset_index(drop=True)
    for column in ("image_path", "source_group", "part"):
        expected[column] = expected[column].astype(str)
        actual[column] = actual[column].astype(str)
    if not expected.equals(actual):
        raise AssertionError("Anonymous-test foreground index geometry changed")
    if (
        len(index) != 1641
        or index.sample_index.nunique() != 1641
        or not index.image_path.astype(str).str.startswith("test/").all()
        or sha256_file(index_path) != EXPECTED_TEST_MASK_INDEX_SHA256
    ):
        raise AssertionError("Anonymous-test foreground index changed")
    valid = index.valid.astype(bool)
    valid_index = index.loc[valid]
    if (
        int(valid.sum()) != EXPECTED_TEST_VALID
        or not valid_index.score.between(0.85, 1.0).all()
        or not valid_index.area.between(0.10, 0.90).all()
        or not set(valid_index.method.astype(str)).issubset({"center", "box_center"})
    ):
        raise AssertionError("Anonymous-test foreground gates changed")
    for row in index.itertuples(index=False):
        path = mask_root / "masks" / f"{int(row.sample_index):06d}.png"
        if bool(row.valid):
            if not path.is_file() or sha256_file(path) != str(row.mask_sha256):
                raise AssertionError(f"Anonymous-test foreground mask changed: {path}")
        elif path.exists():
            raise AssertionError(f"Invalid anonymous-test row has mask file: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_expected = {
        "scope": "anonymous_test",
        "fold": None,
        "rows": 1641,
        "valid": EXPECTED_TEST_VALID,
        "index_sha256": EXPECTED_TEST_MASK_INDEX_SHA256,
        "manifest_sha256": (
            "0470a4bb4630061ef06e92c50a74cb15ba457e476714294872ec426239a1ee6f"
        ),
        "template_sha256": EXPECTED_TEMPLATE_SHA256,
        "sam2_checkpoint_sha256": EXPECTED_SAM2_CHECKPOINT_SHA256,
        "sam2_config_sha256": EXPECTED_SAM2_CONFIG_SHA256,
        "sam2_source_commit": EXPECTED_SAM2_SOURCE_COMMIT,
        "generator_source_sha256": EXPECTED_TEST_MASK_GENERATOR_SHA256,
    }
    for key, value in summary_expected.items():
        if summary.get(key) != value:
            raise AssertionError(
                f"Anonymous-test foreground summary {key}={summary.get(key)!r}, "
                f"expected {value!r}"
            )
    metadata = {
        "root": str(mask_root),
        "index_sha256": EXPECTED_TEST_MASK_INDEX_SHA256,
        "summary_sha256": sha256_file(summary_path),
        "rows": 1641,
        "valid": EXPECTED_TEST_VALID,
        "mask_hashes_verified": True,
    }
    return index.sort_values("sample_index").reset_index(drop=True), metadata


def replay_sampler_exposure(
    sampler: Any,
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    counts = np.zeros(3, dtype=np.int64)
    batch_shapes: dict[str, int] = {}
    for epoch in range(TRAIN_EPOCHS):
        sampler.set_epoch(epoch)
        for selected in sampler:
            batch_counts = np.bincount(
                manifest.iloc[selected].part_index.to_numpy(dtype=np.int64),
                minlength=3,
            )
            if len(selected) != IDENTITIES_PER_BATCH * IMAGES_PER_IDENTITY:
                raise AssertionError("Balanced sampler changed batch size")
            counts += batch_counts
            shape = "/".join(str(int(value)) for value in batch_counts)
            batch_shapes[shape] = batch_shapes.get(shape, 0) + 1
    sampler.set_epoch(0)
    return {
        "epochs": TRAIN_EPOCHS,
        "counts": counts.tolist(),
        "fractions": (counts / counts.sum()).tolist(),
        "batch_part_count_histogram": dict(sorted(batch_shapes.items())),
    }


def materialize_test_part_views(
    test_frame: pd.DataFrame,
    competition_root: Path,
    mask_root: Path,
    mask_index: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    lookup = mask_index.set_index("sample_index").to_dict("index")
    inference_frame = test_frame.copy()
    records: list[dict[str, Any]] = []
    materialized = 0
    for position, row in enumerate(test_frame.itertuples(index=False)):
        foreground = lookup[int(row.sample_index)]
        should_materialize = int(row.part_index) in {1, 2} and bool(
            foreground["valid"]
        )
        inference_path = str(row.image_path)
        image_sha256 = ""
        if should_materialize:
            source_path = competition_root / str(row.image_path)
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            mask_path = mask_root / "masks" / f"{int(row.sample_index):06d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or image.shape[:2] != mask.shape:
                raise AssertionError("Anonymous-test image/mask geometry changed")
            view = foreground_safe_background_blur(image, mask > 127, 0.035)
            target = output_dir / f"{int(row.sample_index):06d}.png"
            if not cv2.imwrite(str(target), view):
                raise RuntimeError(f"Could not write deterministic SAM view: {target}")
            inference_path = str(target.resolve())
            inference_frame.at[position, "image_path"] = inference_path
            image_sha256 = sha256_file(target)
            materialized += 1
        records.append(
            {
                "sample_index": int(row.sample_index),
                "image_path": str(row.image_path),
                "part": str(row.part),
                "materialized_body_view": should_materialize,
                "inference_path": inference_path,
                "inference_image_sha256": image_sha256,
            }
        )
    if materialized != EXPECTED_TEST_BODY_VIEWS:
        raise AssertionError(
            f"Expected {EXPECTED_TEST_BODY_VIEWS} deterministic body views, got "
            f"{materialized}"
        )
    view_index_path = output_dir.parent / "test_view_index.csv"
    pd.DataFrame(records).to_csv(view_index_path, index=False, encoding="utf-8")
    return inference_frame, {
        "route": "head original; valid body fixed SAM background blur; invalid body original",
        "sigma_fraction": 0.035,
        "materialized_body_views": materialized,
        "view_index": str(view_index_path),
        "view_index_sha256": sha256_file(view_index_path),
    }


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
    train_names = set(manifest.image_path.astype(str).map(Path).map(lambda path: path.name))
    overlap = train_names.intersection(template.image_id.astype(str))
    if overlap:
        raise AssertionError(f"Train/test filename overlap: {len(overlap)}")

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
        "full_epoch_rule": "V4.2 fold-0 non-TTA best human epoch 32 fixed before test inference",
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
    records = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
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
    zip_path = args.output_dir / "V4.2-full-noTTA.zip"
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
