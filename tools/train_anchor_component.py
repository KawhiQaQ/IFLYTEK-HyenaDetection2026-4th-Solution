#!/usr/bin/env python3
"""Train and extract one locked anchor component.

Every component starts from its declared legal public/external initializer.
The temporary full checkpoint is hashed and deleted only after the aligned
anonymous-test raw score and prediction artifacts pass their internal gates.
"""

from __future__ import annotations

import argparse
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
    save_checkpoint,
    set_seed,
    train_one_epoch,
    write_json,
)
from model import build_model
from train_cv import load_external_reid_representation
from train_full import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PART_ROWS,
    make_test_frame,
    sha256_file,
)
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
IDENTITIES_PER_BATCH = 16
IMAGES_PER_IDENTITY = 4
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
EXTERNAL_REID_SHA256 = "1dfabf91e0b217f0f7dcf7c2881440a60f199a95868c0fc7c145293aa8876c69"

COMPONENTS: dict[str, dict[str, Any]] = {
    "H": {
        "name": "V4.1 head observation specialist",
        "variant": "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2",
        "epochs": 28,
        "sampler": "cross_part",
        "foreground_mode": "sam_part_view",
        "freeze_blocks": 24,
        "freeze_stages": 2,
        "external_reid": False,
        "cv_score": 0.6864017571459756,
    },
    "A": {
        "name": "V4.3 incumbent",
        "variant": "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
        "epochs": 37,
        "sampler": "balanced_cross_part",
        "foreground_mode": "sam_part_view",
        "freeze_blocks": 24,
        "freeze_stages": 2,
        "external_reid": False,
        "cv_score": 0.716830870686859,
    },
    "B": {
        "name": "V4.8 centered foreground token",
        "variant": "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken",
        "epochs": 36,
        "sampler": "balanced_cross_part",
        "foreground_mode": "sam_part_view",
        "freeze_blocks": 24,
        "freeze_stages": 2,
        "external_reid": False,
        "cv_score": 0.710131541603504,
    },
    "Q": {
        "name": "V9.1 source-aware queue2048",
        "variant": "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048",
        "epochs": 32,
        "sampler": "balanced_cross_part",
        "foreground_mode": "sam_part_view",
        "freeze_blocks": 24,
        "freeze_stages": 2,
        "external_reid": False,
        "cv_score": 0.7183865106101623,
    },
    "D": {
        "name": "V9.3 legal DogFaceNet representation transfer",
        "variant": "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
        "epochs": 33,
        "sampler": "balanced_cross_part",
        "foreground_mode": "sam_part_view",
        "freeze_blocks": 24,
        "freeze_stages": 2,
        "external_reid": True,
        "cv_score": 0.7157887522746625,
    },
    "C": {
        "name": "V3.1 DINOv3 ConvNeXt-L dual-level",
        "variant": "dinov3_convnext_large_dual_mgn_cov",
        "epochs": 24,
        "sampler": "cross_part",
        "foreground_mode": "sam_bg",
        "freeze_blocks": 2,
        "freeze_stages": 2,
        "external_reid": False,
        "cv_score": 0.6223013697417694,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", required=True, choices=sorted(COMPONENTS))
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--train-foreground-mask-root", type=Path, required=True)
    parser.add_argument("--test-foreground-mask-root", type=Path, required=True)
    parser.add_argument("--external-reid-checkpoint", type=Path)
    parser.add_argument("--external-reid-audit", type=Path)
    parser.add_argument(
        "--expected-external-reid-sha256",
        default=EXTERNAL_REID_SHA256,
        help=(
            "Expected SHA-256 for the DogFace Stage-A checkpoint. The default "
            "is the exact checkpoint used for the submitted result; override "
            "only when Stage A was independently retrained with the included code."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    source_dir = ROOT / "src/hyenaid"
    paths = {
        "train_anchor_component.py": Path(__file__).resolve(),
        "model.py": source_dir / "model.py",
        "data.py": source_dir / "data.py",
        "engine.py": source_dir / "engine.py",
        "losses.py": source_dir / "losses.py",
        "train_cv.py": source_dir / "train_cv.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def replay_sampler_exposure(
    sampler: Any, manifest: pd.DataFrame, epochs: int
) -> dict[str, object]:
    counts = np.zeros(3, dtype=np.int64)
    shapes: dict[str, int] = {}
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for selected in sampler:
            batch_counts = np.bincount(
                manifest.iloc[selected].part_index.to_numpy(dtype=np.int64),
                minlength=3,
            )
            if len(selected) != IDENTITIES_PER_BATCH * IMAGES_PER_IDENTITY:
                raise AssertionError("full-data sampler changed batch size")
            counts += batch_counts
            shape = "/".join(str(int(value)) for value in batch_counts)
            shapes[shape] = shapes.get(shape, 0) + 1
    sampler.set_epoch(0)
    return {
        "epochs": epochs,
        "counts": counts.tolist(),
        "fractions": (counts / counts.sum()).tolist(),
        "batch_part_count_histogram": dict(sorted(shapes.items())),
    }


def validate_component_args(args: argparse.Namespace, spec: dict[str, Any]) -> None:
    external_values = (args.external_reid_checkpoint, args.external_reid_audit)
    if spec["external_reid"]:
        if any(value is None for value in external_values):
            raise ValueError("D requires both legal external ReID files")
        for value in external_values:
            assert value is not None
            if not value.is_file():
                raise FileNotFoundError(value)
    elif any(value is not None for value in external_values):
        raise ValueError(f"component {args.component} forbids external ReID files")


def validate_full_inputs_deferred_test_files(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Validate labels/template now; require test pixels only at extraction.

    This lets the four-hour all-data training overlap a byte-for-byte migration
    of the official anonymous test directory. Test pixels still never enter a
    training loader and are required in full before the sole inference pass.
    """
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("unexpected manifest hash")
    manifest = pd.read_csv(args.manifest)
    if (
        len(manifest) != 4067
        or manifest.fold.nunique() != 5
        or manifest.sample_index.nunique() != 4067
        or manifest.image_path.astype(str).str.startswith("test/").any()
    ):
        raise AssertionError("unexpected full training manifest")
    labels = (
        manifest[["individual_id", "label_index"]]
        .drop_duplicates()
        .sort_values("label_index")
        .individual_id.astype(str)
        .tolist()
    )
    if len(labels) != 255 or sorted(manifest.label_index.unique()) != list(range(255)):
        raise AssertionError("training label boundary changed")
    template = pd.read_csv(args.template)
    if list(template.columns) != ["image_id", "predicted_id"]:
        raise AssertionError("unexpected template columns")
    if len(template) != 1641 or template.image_id.nunique() != 1641:
        raise AssertionError("unexpected template rows")
    test_frame = make_test_frame(template)
    if test_frame.part.value_counts().to_dict() != EXPECTED_PART_ROWS:
        raise AssertionError("unexpected test part counts")
    return manifest, template, test_frame, labels


def main() -> None:
    args = parse_args()
    for name in (
        "competition_root",
        "manifest",
        "template",
        "train_foreground_mask_root",
        "test_foreground_mask_root",
        "output_dir",
        "external_reid_checkpoint",
        "external_reid_audit",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.workers < 0 or args.eval_batch_size < 1:
        raise ValueError("invalid loader resources")
    spec = COMPONENTS[args.component]
    validate_component_args(args, spec)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validate_model_augmentation_profile(spec["variant"], "arbase_source_jitter")
    manifest, template, test_frame, labels = validate_full_inputs_deferred_test_files(args)
    if sha256_file(args.template) != EXPECTED_TEMPLATE_SHA256:
        raise AssertionError("official template SHA-256 changed")
    if manifest.image_path.astype(str).str.startswith("test/").any():
        raise AssertionError("anonymous test entered full training")
    train_names = set(manifest.image_path.astype(str).map(Path).map(lambda x: x.name))
    if train_names.intersection(template.image_id.astype(str)):
        raise AssertionError("train/test filename overlap")

    geometry_stats = (
        None
        if args.component == "C"
        else build_crop_geometry_stats(manifest, args.competition_root)
    )
    if geometry_stats is not None and geometry_stats["rows"] != 4067:
        raise AssertionError("full-data geometry stats changed")
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
        augmentation_profile="arbase_source_jitter",
        sampler_profile=spec["sampler"],
        geometry_stats=geometry_stats,
        foreground_mask_root=args.train_foreground_mask_root,
        foreground_mode=spec["foreground_mode"],
    )
    if len(train_loader) != 64:
        raise AssertionError(f"expected 64 full-data steps, got {len(train_loader)}")
    sampler_exposure = replay_sampler_exposure(
        train_sampler, manifest, int(spec["epochs"])
    )

    device = torch.device("cuda")
    model = build_model(
        spec["variant"],
        num_classes=len(labels),
        image_size=IMAGE_SIZE,
        embedding_dim=EMBEDDING_DIM,
        local_queries=LOCAL_QUERIES,
        pretrained=not spec["external_reid"],
        arc_scale=ARC_SCALE,
        arc_margin=ARC_MARGIN,
        part_delta_scale=PART_DELTA_SCALE,
        freeze_blocks=int(spec["freeze_blocks"]),
        freeze_stages=int(spec["freeze_stages"]),
        grad_checkpointing=True,
    )
    external_metadata: dict[str, Any] = {}
    if spec["external_reid"]:
        assert args.external_reid_checkpoint is not None
        assert args.external_reid_audit is not None
        external_metadata = load_external_reid_representation(
            model,
            args.external_reid_checkpoint,
            args.external_reid_audit,
            args.expected_external_reid_sha256,
        )
    model = model.to(device)
    set_model_metadata(model, manifest, len(labels))
    if args.component == "Q":
        if not model.training_instance_queue or model.instance_queue_capacity != 2048:
            raise AssertionError("Q queue contract changed")
    elif getattr(model, "training_instance_queue", False):
        raise AssertionError(f"unexpected queue in {args.component}")

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
        "version": "V10.3-full",
        "component": args.component,
        "component_name": spec["name"],
        "competition_root": str(args.competition_root),
        "manifest": str(args.manifest),
        "template": str(args.template),
        "output_dir": str(args.output_dir),
        "model_variant": spec["variant"],
        "model_name": model.model_name,
        "pretraining_source": model.pretraining_source,
        "init_checkpoint": None,
        "competition_checkpoint_loaded": False,
        "external_reid_initialization": external_metadata,
        "augmentation_profile": "arbase_source_jitter",
        "sampler_profile": spec["sampler"],
        "sampler_exposure": sampler_exposure,
        "foreground_mode": spec["foreground_mode"],
        "train_foreground_mask_artifact": train_foreground_metadata,
        "train_foreground_summary": train_foreground_summary,
        "test_foreground_mask_artifact": test_foreground_metadata,
        "released_crop_geometry_stats": geometry_stats,
        "train_epochs": int(spec["epochs"]),
        "scheduler_horizon_epochs": SCHEDULER_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "image_size": IMAGE_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "local_queries": LOCAL_QUERIES,
        "identities_per_batch": IDENTITIES_PER_BATCH,
        "images_per_identity": IMAGES_PER_IDENTITY,
        "freeze_blocks": int(spec["freeze_blocks"]),
        "freeze_stages": int(spec["freeze_stages"]),
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
        "supcon_temperature": SUPCON_TEMPERATURE,
        "triplet_scale": TRIPLET_SCALE,
        "grad_clip": GRAD_CLIP,
        "optimizer": "adamw",
        "amp_initial_scale": 1024.0,
        "seed": SEED,
        "tta_flip": False,
        "gallery_decoder": False,
        "full_epoch_rule": (
            f"component {args.component} fold0 selected human epoch "
            f"{spec['epochs']} fixed before test inference"
        ),
        "cv_reference_non_tta": spec["cv_score"],
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
    for epoch in range(int(spec["epochs"])):
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
            raise FloatingPointError("full component failed numerical/memory gate")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"component={args.component} epoch={epoch + 1:02d}/{spec['epochs']} "
            f"loss={losses['loss']:.4f} part_ce={losses['part_ce']:.4f} "
            f"shared_ce={losses['shared_ce']:.4f} time={record['seconds']:.1f}s "
            f"mem={record['gpu_peak_gb']:.1f}GB",
            flush=True,
        )

    checkpoint_path = args.output_dir / "model.pt"
    save_checkpoint(
        {
            "model": model.state_dict(),
            "epoch": int(spec["epochs"]) - 1,
            "labels": labels,
            "config": config,
        },
        checkpoint_path,
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    log_records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if [int(row["epoch"]) for row in log_records] != list(range(int(spec["epochs"]))):
        raise AssertionError("non-contiguous full component training log")

    missing_test = [
        path
        for path in test_frame.image_path.astype(str)
        if not (args.competition_root / path).is_file()
    ]
    if missing_test:
        raise FileNotFoundError(
            f"official test migration incomplete before extraction: {missing_test[0]}"
        )
    eval_foreground = spec["foreground_mode"] == "sam_part_view"
    return_foreground_mask = args.component == "B"
    test_loader = make_eval_loader(
        test_frame,
        args.competition_root,
        IMAGE_SIZE,
        args.eval_batch_size,
        args.workers,
        SEED,
        augmentation_profile="arbase_source_jitter",
        geometry_stats=geometry_stats,
        foreground_mask_root=(args.test_foreground_mask_root if eval_foreground else None),
        foreground_mode=("sam_part_view" if eval_foreground else "none"),
        return_foreground_mask=return_foreground_mask,
        validated_foreground_index=(test_mask_index if eval_foreground else None),
    )
    features = extract(
        model,
        test_loader,
        device,
        tta_flip=False,
        return_local=False,
        local_grid=6,
    )
    scores = features["classifier_score"].float().cpu()
    sample_index = features["sample_index"].cpu()
    part_index = features["part_index"].cpu()
    if scores.shape != (1641, 255):
        raise AssertionError(f"unexpected test score shape: {tuple(scores.shape)}")
    if not torch.equal(sample_index, torch.arange(1641)):
        raise AssertionError("test score order changed")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("non-finite test score")

    score_path = args.output_dir / "component_scores.pt"
    torch.save(
        {
            "version": "V10.3-full",
            "component": args.component,
            "sample_index": sample_index,
            "part_index": part_index,
            "label_order": labels,
            "classifier_score": scores,
            "provenance": {
                "checkpoint_sha256": checkpoint_sha,
                "train_epochs": int(spec["epochs"]),
                "init_checkpoint": None,
                "competition_checkpoint_loaded": False,
                "external_reid_initialization": external_metadata,
            },
        },
        score_path,
    )
    predicted = prediction_frame(test_frame, features, scores, labels)
    prediction_path = args.output_dir / "test_predictions.csv"
    predicted.to_csv(prediction_path, index=False, encoding="utf-8")
    expected_ids = np.asarray(labels, dtype=object)[scores.argmax(1).numpy()].astype(str)
    if not np.array_equal(predicted.predicted_id.astype(str).to_numpy(), expected_ids):
        raise AssertionError("score argmax/prediction mismatch")

    report = {
        "status": "PASS",
        "version": "V10.3-full",
        "component": args.component,
        "component_name": spec["name"],
        "elapsed_seconds": time.time() - started,
        "train_epochs": int(spec["epochs"]),
        "scheduler_horizon_epochs": SCHEDULER_EPOCHS,
        "train_rows": len(manifest),
        "test_rows": len(test_frame),
        "tta_flip": False,
        "decoder": "classifier_raw_score",
        "test_observation": (
            "head original; body deterministic SAM view"
            if eval_foreground
            else "ordinary raw organizer crop"
        ),
        "foreground_mask_returned_to_model": return_foreground_mask,
        "manifest_sha256": config["manifest_sha256"],
        "template_sha256": config["template_sha256"],
        "train_foreground_index_sha256": EXPECTED_TRAIN_MASK_INDEX_SHA256,
        "test_foreground_index_sha256": EXPECTED_TEST_MASK_INDEX_SHA256,
        "source_sha256": config["source_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": sha256_file(args.output_dir / "config.json"),
        "train_log_sha256": sha256_file(log_path),
        "component_scores_sha256": sha256_file(score_path),
        "test_predictions_sha256": sha256_file(prediction_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_retained": True,
    }
    report_path = args.output_dir / "component_report.json"
    write_json(report, report_path)

    checkpoint_path.unlink()
    if checkpoint_path.exists():
        raise AssertionError("temporary component checkpoint was not deleted")
    cleanup = {
        "status": "PASS",
        "component": args.component,
        "deleted_checkpoint": str(checkpoint_path),
        "deleted_checkpoint_sha256": checkpoint_sha,
        "reason": "V10.3-full sequential 8.6GB disk lifecycle after verified raw-score extraction",
        "recoverability": "retrain exactly from the declared legal initializer and fixed recipe",
    }
    write_json(cleanup, args.output_dir / "checkpoint_cleanup.json")
    report["checkpoint_retained"] = False
    report["checkpoint_cleanup_sha256"] = sha256_file(
        args.output_dir / "checkpoint_cleanup.json"
    )
    write_json(report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
