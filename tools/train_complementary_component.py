#!/usr/bin/env python3
"""Train one complementary all-data component and extract test raw scores.

The trained model never leaves memory: the server cannot safely hold an 8 GB
temporary checkpoint.  A deterministic tensor-wise state hash is recorded
before the sole non-TTA anonymous-test pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
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
    set_seed,
    train_one_epoch,
    write_json,
)
from model import build_model
from train_full import EXPECTED_MANIFEST_SHA256, make_test_frame, sha256_file
from train_full_v4_2 import (
    EXPECTED_TEMPLATE_SHA256,
    EXPECTED_TEST_MASK_INDEX_SHA256,
    EXPECTED_TRAIN_MASK_INDEX_SHA256,
    set_model_metadata,
    validate_test_mask_artifact,
    validate_train_mask_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260719
SCHEDULER_EPOCHS = 48
WARMUP_EPOCHS = 3
IMAGE_SIZE = 448
EMBEDDING_DIM = 512
LOCAL_QUERIES = 4
IDS_PER_BATCH = 16
IMAGES_PER_ID = 4
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
BIOCLIP_SHA256 = "e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4"

COMPONENTS: dict[str, dict[str, Any]] = {
    "P": {
        "name": "V2.63-P8 generic DINOv3 background-debiased training",
        "variant": "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2",
        "epochs": 18,
        "augmentation": "arbase_source_jitter",
        "sampler": "cross_part",
        "foreground_mode": "sam_bg",
        "freeze_blocks": 24,
        "backbone_lr": 2.4e-5,
        "eval_batch_size": 20,
        "geometry": True,
        "queue": False,
        "external_generic": False,
        "cv_score": 0.6855191601978851,
    },
    "F": {
        "name": "V10.4 legal generic BioCLIP-B visual component",
        "variant": "bioclip_vitb_patch_mgn_cov_ptoposupcon_queue2048",
        "epochs": 14,
        "augmentation": "arbase_clip",
        "sampler": "balanced_cross_part",
        "foreground_mode": "sam_part_view",
        "freeze_blocks": 8,
        "backbone_lr": 3.0e-5,
        "eval_batch_size": 64,
        "geometry": False,
        "queue": True,
        "external_generic": True,
        "cv_score": 0.34606708450503615,
    },
    "K": {
        "name": "V11.3 body-only deterministic quality view",
        "variant": "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048",
        "epochs": 20,
        "augmentation": "arbase_source_jitter_quality_body",
        "sampler": "balanced_cross_part",
        "foreground_mode": "sam_part_view",
        "freeze_blocks": 24,
        "backbone_lr": 2.4e-5,
        "eval_batch_size": 20,
        "geometry": True,
        "queue": True,
        "external_generic": False,
        "cv_score": 0.6951782960144968,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=sorted(COMPONENTS), required=True)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--train-foreground-mask-root", type=Path, required=True)
    parser.add_argument("--test-foreground-mask-root", type=Path, required=True)
    parser.add_argument("--bioclip-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    paths = {
        "trainer": Path(__file__).resolve(),
        "model.py": ROOT / "src/hyenaid/model.py",
        "data.py": ROOT / "src/hyenaid/data.py",
        "engine.py": ROOT / "src/hyenaid/engine.py",
        "losses.py": ROOT / "src/hyenaid/losses.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        # Flatten first: PyTorch does not allow a 0-D scalar tensor (for
        # example BatchNorm's integer num_batches_tracked buffer) to be
        # reinterpreted as bytes when the element sizes differ.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("manifest hash changed")
    if sha256_file(args.template) != EXPECTED_TEMPLATE_SHA256:
        raise AssertionError("template hash changed")
    manifest = pd.read_csv(args.manifest)
    template = pd.read_csv(args.template)
    if (
        len(manifest) != 4067
        or manifest.sample_index.nunique() != 4067
        or manifest.image_path.astype(str).str.startswith("test/").any()
        or len(template) != 1641
        or template.image_id.nunique() != 1641
        or list(template.columns) != ["image_id", "predicted_id"]
    ):
        raise AssertionError("train/test boundary changed")
    if set(manifest.image_path.astype(str).map(Path).map(lambda path: path.name)) & set(
        template.image_id.astype(str)
    ):
        raise AssertionError("train/test filename overlap")
    labels = (
        manifest[["individual_id", "label_index"]]
        .drop_duplicates()
        .sort_values("label_index")
        .individual_id.astype(str)
        .tolist()
    )
    if len(labels) != 255:
        raise AssertionError("label space changed")
    test = make_test_frame(template)
    if test.part.value_counts().to_dict() != {
        "head": 798,
        "left_body": 414,
        "right_body": 429,
    }:
        raise AssertionError("test part geometry changed")
    return manifest, template, test, labels


def main() -> None:
    args = parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    spec = COMPONENTS[args.component]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if spec["external_generic"]:
        if args.bioclip_checkpoint is None or not args.bioclip_checkpoint.is_file():
            raise FileNotFoundError(args.bioclip_checkpoint)
        if sha256_file(args.bioclip_checkpoint) != BIOCLIP_SHA256:
            raise AssertionError("BioCLIP checkpoint changed")
    elif args.bioclip_checkpoint is not None:
        raise ValueError(f"{args.component} forbids external generic checkpoint")

    validate_model_augmentation_profile(spec["variant"], spec["augmentation"])
    manifest, template, test_frame, labels = validate_inputs(args)
    _, train_mask_metadata = validate_foreground_mask_artifact(
        manifest, args.train_foreground_mask_root, verify_mask_hashes=True
    )
    train_mask_summary = validate_train_mask_provenance(args.train_foreground_mask_root)
    test_mask_index, test_mask_metadata = validate_test_mask_artifact(
        test_frame, args.test_foreground_mask_root
    )
    geometry = (
        build_crop_geometry_stats(manifest, args.competition_root)
        if spec["geometry"]
        else None
    )
    if geometry is not None and geometry["rows"] != 4067:
        raise AssertionError("geometry used non-training rows")

    set_seed(SEED)
    train_loader, sampler = make_train_loader(
        manifest,
        args.competition_root,
        IMAGE_SIZE,
        args.workers,
        IDS_PER_BATCH,
        IMAGES_PER_ID,
        SEED,
        augmentation_profile=spec["augmentation"],
        sampler_profile=spec["sampler"],
        geometry_stats=geometry,
        foreground_mask_root=args.train_foreground_mask_root,
        foreground_mode=spec["foreground_mode"],
    )
    if len(train_loader) != 64:
        raise AssertionError("full-data step count changed")
    device = torch.device("cuda")
    model = build_model(
        spec["variant"],
        num_classes=255,
        image_size=IMAGE_SIZE,
        embedding_dim=EMBEDDING_DIM,
        local_queries=LOCAL_QUERIES,
        pretrained=True,
        arc_scale=ARC_SCALE,
        arc_margin=ARC_MARGIN,
        part_delta_scale=PART_DELTA_SCALE,
        freeze_blocks=spec["freeze_blocks"],
        freeze_stages=2,
        grad_checkpointing=True,
        external_pretrained_path=(
            str(args.bioclip_checkpoint) if spec["external_generic"] else None
        ),
    ).to(device)
    set_model_metadata(model, manifest, 255)
    if bool(getattr(model, "training_instance_queue", False)) != bool(spec["queue"]):
        raise AssertionError("queue contract changed")
    optimizer = build_optimizer(
        model,
        spec["backbone_lr"],
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
        "version": "V11.8-full",
        "component": args.component,
        "component_name": spec["name"],
        "model_variant": spec["variant"],
        "pretraining_source": model.pretraining_source,
        "external_generic_checkpoint_sha256": (
            BIOCLIP_SHA256 if spec["external_generic"] else None
        ),
        "init_checkpoint": None,
        "competition_checkpoint_loaded": False,
        "train_epochs": spec["epochs"],
        "scheduler_horizon_epochs": SCHEDULER_EPOCHS,
        "full_epoch_rule": (
            f"fold0 selected human epoch {spec['epochs']} fixed before full training"
        ),
        "augmentation_profile": spec["augmentation"],
        "sampler_profile": spec["sampler"],
        "foreground_mode": spec["foreground_mode"],
        "train_foreground_mask_artifact": train_mask_metadata,
        "train_foreground_summary": train_mask_summary,
        "test_foreground_mask_artifact": test_mask_metadata,
        "geometry_stats": geometry,
        "image_size": IMAGE_SIZE,
        "identities_per_batch": IDS_PER_BATCH,
        "images_per_identity": IMAGES_PER_ID,
        "freeze_blocks": spec["freeze_blocks"],
        "backbone_lr": spec["backbone_lr"],
        "head_lr": HEAD_LR,
        "layer_decay": LAYER_DECAY,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "tta_flip": False,
        "manifest_sha256": sha256_file(args.manifest),
        "template_sha256": sha256_file(args.template),
        "source_sha256": source_hashes(),
        "train_rows": 4067,
        "test_rows": 1641,
        "cv_reference_non_tta": spec["cv_score"],
        "checkpoint_policy": "in-memory only; deterministic tensor-wise state hash",
    }
    write_json(config, args.output_dir / "config.json")
    log_path = args.output_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    started = time.time()
    for epoch in range(spec["epochs"]):
        epoch_started = time.time()
        torch.cuda.reset_peak_memory_stats()
        sampler.set_epoch(epoch)
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
            raise FloatingPointError(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"component={args.component} epoch={epoch + 1:02d}/{spec['epochs']} "
            f"loss={record['loss']:.4f} time={record['seconds']:.1f}s "
            f"mem={record['gpu_peak_gb']:.2f}GB",
            flush=True,
        )
    model_hash = state_sha256(model)
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    if [row["epoch"] for row in records] != list(range(spec["epochs"])):
        raise AssertionError("training lifecycle changed")

    uses_test_mask = spec["foreground_mode"] == "sam_part_view"
    test_loader = make_eval_loader(
        test_frame,
        args.competition_root,
        IMAGE_SIZE,
        spec["eval_batch_size"],
        args.workers,
        SEED,
        augmentation_profile=spec["augmentation"],
        geometry_stats=geometry,
        foreground_mask_root=(args.test_foreground_mask_root if uses_test_mask else None),
        foreground_mode=("sam_part_view" if uses_test_mask else "none"),
        validated_foreground_index=(test_mask_index if uses_test_mask else None),
    )
    features = extract(model, test_loader, device, tta_flip=False, return_local=False)
    scores = features["classifier_score"].float().cpu()
    sample_index = features["sample_index"].long().cpu()
    part_index = features["part_index"].long().cpu()
    if (
        tuple(scores.shape) != (1641, 255)
        or not torch.equal(sample_index, torch.arange(1641))
        or not torch.isfinite(scores).all()
        or not torch.equal(part_index, torch.as_tensor(test_frame.part_index.to_numpy()))
    ):
        raise AssertionError("test score contract changed")
    score_path = args.output_dir / "component_scores.pt"
    torch.save(
        {
            "version": "V11.8-full",
            "component": args.component,
            "sample_index": sample_index,
            "part_index": part_index,
            "label_order": labels,
            "classifier_score": scores,
            "provenance": {
                "in_memory_model_state_sha256": model_hash,
                "train_epochs": spec["epochs"],
                "init_checkpoint": None,
                "competition_checkpoint_loaded": False,
            },
        },
        score_path,
    )
    predictions = prediction_frame(test_frame, features, scores, labels)
    prediction_path = args.output_dir / "test_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    expected = np.asarray(labels, dtype=object)[scores.argmax(1).numpy()].astype(str)
    if not np.array_equal(predictions.predicted_id.astype(str).to_numpy(), expected):
        raise AssertionError("score/prediction mismatch")
    report = {
        "status": "PASS",
        "version": "V11.8-full",
        "component": args.component,
        "component_name": spec["name"],
        "train_epochs": spec["epochs"],
        "train_rows": 4067,
        "test_rows": 1641,
        "tta_flip": False,
        "anonymous_test_train_rows": 0,
        "competition_checkpoint_loaded": False,
        "in_memory_model_state_sha256": model_hash,
        "checkpoint_written": False,
        "elapsed_seconds": time.time() - started,
        "component_scores_sha256": sha256_file(score_path),
        "test_predictions_sha256": sha256_file(prediction_path),
        "config_sha256": sha256_file(args.output_dir / "config.json"),
        "train_log_sha256": sha256_file(log_path),
        "train_mask_index_sha256": EXPECTED_TRAIN_MASK_INDEX_SHA256,
        "test_mask_index_sha256": EXPECTED_TEST_MASK_INDEX_SHA256,
    }
    write_json(report, args.output_dir / "component_report.json")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
