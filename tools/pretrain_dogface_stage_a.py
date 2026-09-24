from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


CORE_SRC = Path(__file__).resolve().parents[1] / "src" / "hyenaid"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from data import IdentityPartBatchSampler, build_transform, seed_worker  # noqa: E402
from engine import (  # noqa: E402
    build_optimizer,
    build_scheduler,
    set_seed,
    train_one_epoch,
)
from model import build_model  # noqa: E402


MODEL_VARIANT = "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon"
DATASET_NAME = "DogFaceNet_224resized"
EXPECTED_ROWS = 8_363
EXPECTED_IDENTITIES = 1_393
SEED = 20_260_719
IMAGE_SIZE = 448
EPOCHS = 16
WARMUP_EPOCHS = 3
FREEZE_BLOCKS = 24
BACKBONE_LR = 2.4e-5
HEAD_LR = 3e-4
LAYER_DECAY = 0.82
WEIGHT_DECAY = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DogFaceDataset(Dataset[tuple[torch.Tensor, int, int, int, int, torch.Tensor]]):
    def __init__(self, frame: pd.DataFrame, root: Path) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.root = root.resolve()
        self.transform = build_transform(IMAGE_SIZE, True, profile="arbase_head")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.root / str(row.image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        transformed = self.transform(image=image)["image"]
        return (
            transformed,
            int(row.label_index),
            0,
            int(row.sample_index),
            int(row.source_code),
            torch.zeros(2, dtype=torch.float32),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_inputs(
    manifest_path: Path, audit_path: Path, data_root: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required = {
        "status": "PASS",
        "dataset": DATASET_NAME,
        "license": "CC-BY-4.0",
        "hyena_or_hyenaid_present": False,
        "wildbook_or_competition_source_present": False,
        "images": EXPECTED_ROWS,
        "identities": EXPECTED_IDENTITIES,
        "anonymous_test_rows_in_manifest": 0,
    }
    for key, expected in required.items():
        if audit.get(key) != expected:
            raise AssertionError(f"Dataset audit {key}={audit.get(key)!r} != {expected!r}")
    if audit.get("competition_hash_audit", {}).get("exact_pixel_overlap") != 0:
        raise AssertionError("External images overlap official decoded pixels")
    if audit.get("manifest_sha256") != sha256_file(manifest_path):
        raise AssertionError("External manifest hash differs from audit")
    frame = pd.read_csv(manifest_path)
    required_columns = {
        "sample_index",
        "external_dataset",
        "identity_name",
        "label_index",
        "part",
        "part_index",
        "source_code",
        "source_group",
        "image_path",
        "competition_exact_pixel_overlap",
    }
    if not required_columns.issubset(frame.columns):
        raise AssertionError(f"External manifest columns changed: {frame.columns.tolist()}")
    if len(frame) != EXPECTED_ROWS or frame.label_index.nunique() != EXPECTED_IDENTITIES:
        raise AssertionError("External manifest dimensions changed")
    if set(frame.external_dataset.astype(str)) != {DATASET_NAME}:
        raise AssertionError("External manifest contains another dataset")
    if set(frame.part_index.astype(int)) != {0} or set(frame.part.astype(str)) != {"head"}:
        raise AssertionError("DogFaceNet must use only the head pretraining route")
    if frame.competition_exact_pixel_overlap.astype(bool).any():
        raise AssertionError("External manifest contains an official pixel overlap")
    if frame.image_path.astype(str).str.startswith(("test/", "hyena/", "/")).any():
        raise AssertionError("Competition or absolute path entered external pretraining")
    if sorted(frame.label_index.astype(int).unique()) != list(range(EXPECTED_IDENTITIES)):
        raise AssertionError("External labels must be contiguous and dataset-local")
    if frame.source_group.astype(str).nunique() != len(frame):
        raise AssertionError("Each aligned DogFaceNet image must be its own source group")
    missing = [
        path
        for path in frame.image_path.astype(str)
        if not (data_root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} external images")
    return frame, audit


def gradient_audit(model: torch.nn.Module) -> dict[str, float | int]:
    adapted = [
        parameter.grad.detach().float().norm()
        for parameter in model.prefix_adaptformer_parameters()
        if parameter.grad is not None
    ]
    final_block = [
        parameter.grad.detach().float().norm()
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.blocks.31.") and parameter.grad is not None
    ]
    frozen_prefix_base = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.blocks.0.")
        and ".adapter_" not in name
        and not parameter.requires_grad
    ]
    frozen_with_grad = sum(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in frozen_prefix_base
    )
    if not adapted or float(torch.stack(adapted).norm()) <= 0.0:
        raise AssertionError("AdaptFormer received no external-pretraining gradient")
    if not final_block or float(torch.stack(final_block).norm()) <= 0.0:
        raise AssertionError("Final DINO block received no external-pretraining gradient")
    if frozen_with_grad:
        raise AssertionError("Frozen prefix base received a gradient")
    return {
        "adaptformer_gradient_norm": float(torch.stack(adapted).norm()),
        "final_block_gradient_norm": float(torch.stack(final_block).norm()),
        "frozen_prefix_base_tensors": len(frozen_prefix_base),
        "frozen_prefix_base_tensors_with_gradient": int(frozen_with_grad),
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest_path = args.manifest.resolve()
    audit_path = args.dataset_audit.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, dataset_audit = validate_inputs(
        manifest_path, audit_path, data_root
    )

    set_seed(SEED)
    dataset = DogFaceDataset(frame, data_root)
    sampler = IdentityPartBatchSampler(
        frame,
        identities_per_batch=16,
        images_per_identity=4,
        seed=SEED,
        same_part_group=False,
        balanced_cross_part_group=False,
    )
    if args.smoke:
        sampler.steps = 1
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = build_model(
        MODEL_VARIANT,
        num_classes=EXPECTED_IDENTITIES,
        image_size=IMAGE_SIZE,
        embedding_dim=512,
        local_queries=4,
        pretrained=True,
        arc_scale=30.0,
        arc_margin=0.20,
        part_delta_scale=0.20,
        freeze_blocks=FREEZE_BLOCKS,
        freeze_stages=2,
        grad_checkpointing=True,
    )
    availability = torch.zeros(3, EXPECTED_IDENTITIES, dtype=torch.bool)
    availability[0] = True
    model.set_part_availability(availability)
    device = torch.device("cuda")
    model = model.to(device)
    optimizer = build_optimizer(
        model,
        BACKBONE_LR,
        HEAD_LR,
        WEIGHT_DECAY,
        LAYER_DECAY,
        optimizer_name="adamw",
    )
    total_epochs = 1 if args.smoke else EPOCHS
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_epochs * len(loader),
        warmup_steps=min(WARMUP_EPOCHS, total_epochs) * len(loader),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    log_path = output_dir / ("smoke_log.jsonl" if args.smoke else "train_log.jsonl")
    log_path.write_text("", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    last_metrics: dict[str, float] | None = None
    for epoch in range(total_epochs):
        sampler.set_epoch(epoch)
        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler,
            device,
            label_smoothing=0.05,
            shared_ce_weight=0.50,
            supcon_weight=0.15,
            triplet_weight=0.30,
            prototype_weight=0.0,
            supcon_temperature=0.10,
            triplet_scale=0.10,
            grad_clip=1.0,
        )
        record = {
            "epoch": epoch + 1,
            **metrics,
            "seconds": time.time() - started,
            "gpu_peak_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        last_metrics = metrics

    gradients = gradient_audit(model)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    if peak_gb >= 24.0:
        raise AssertionError(f"Peak CUDA allocation {peak_gb:.3f} GB >= 24 GB")
    common_report = {
        "status": "PASS",
        "kind": "external_animal_identity_pretraining",
        "dataset": DATASET_NAME,
        "dataset_rows": len(frame),
        "dataset_identities": frame.label_index.nunique(),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_audit_sha256": sha256_file(audit_path),
        "license": dataset_audit["license"],
        "hyena_or_hyenaid_present": False,
        "competition_data_used": False,
        "competition_checkpoint_loaded": False,
        "model_variant": MODEL_VARIANT,
        "generic_pretraining_source": model.pretraining_source,
        "epochs": total_epochs,
        "fixed_final_epoch_export": not args.smoke,
        "last_metrics": last_metrics,
        "elapsed_seconds": time.time() - started,
        "gpu_peak_gb": peak_gb,
        **gradients,
    }
    if args.smoke:
        report_path = output_dir / "smoke_audit.json"
        report_path.write_text(
            json.dumps(common_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(common_report, indent=2, ensure_ascii=False), flush=True)
        return

    checkpoint_path = output_dir / "external_reid_final.pt"
    checkpoint = {
        "kind": "external_animal_identity_pretraining",
        "dataset": DATASET_NAME,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_audit_sha256": sha256_file(audit_path),
        "license": dataset_audit["license"],
        "hyena_or_hyenaid_present": False,
        "competition_data_used": False,
        "competition_checkpoint_loaded": False,
        "model_variant": MODEL_VARIANT,
        "generic_pretraining_source": model.pretraining_source,
        "epochs": EPOCHS,
        "fixed_final_epoch_export": True,
        "labels": (
            frame[["identity_name", "label_index"]]
            .drop_duplicates()
            .sort_values("label_index")
            .identity_name.astype(str)
            .tolist()
        ),
        "model": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "config": {
            "image_size": IMAGE_SIZE,
            "epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "freeze_blocks": FREEZE_BLOCKS,
            "backbone_lr": BACKBONE_LR,
            "head_lr": HEAD_LR,
            "layer_decay": LAYER_DECAY,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "identities_per_batch": 16,
            "images_per_identity": 4,
            "augmentation_profile": "arbase_head",
        },
    }
    torch.save(checkpoint, checkpoint_path)
    common_report["checkpoint"] = str(checkpoint_path)
    common_report["checkpoint_sha256"] = sha256_file(checkpoint_path)
    report_path = output_dir / "checkpoint_audit.json"
    report_path.write_text(
        json.dumps(common_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(common_report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
