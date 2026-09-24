from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import (
    build_crop_geometry_stats,
    build_fold_train_edge_side_counts,
    make_eval_loader,
    make_train_loader,
    validate_foreground_mask_artifact,
    validate_model_augmentation_profile,
)
from engine import (
    ModelEMA,
    ModelSWA,
    PARTS,
    build_optimizer,
    build_scheduler,
    capture_suffix_start_point_anchor,
    classifier_metrics,
    competition_metrics,
    extract,
    gallery_fused_scores,
    prediction_frame,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    write_json,
)
from model import (
    BIOCLIP_PRETRAINING_SHA256,
    DINOV2_GIANT_REGISTER_PRETRAINING_SHA256,
    PETFACE_R50_PRETRAINING_SHA256,
    RADIO_V25_B_PRETRAINING_SHA256,
    SAM2_HIERA_SMALL_PRETRAINING_SHA256,
    TIPS_L14_HR_PRETRAINING_SHA256,
    build_model,
    configure_suffix_stochastic_depth,
)
from bioclip2_vision import BIOCLIP2_PRETRAINING_SHA256


EXPECTED_MANIFEST_SHA256 = (
    "0470a4bb4630061ef06e92c50a74cb15ba457e476714294872ec426239a1ee6f"
)
FROM_GENERIC_ONLY_VARIANTS = {
    "sam2_hiera_small_fpn_mgn_cov_ptoposupcon_queue2048",
    "tips_l14_hr_patch_mgn_cov_ptoposupcon_queue2048",
    "dinov2_giant_reg_patch_mgn_cov_geosie2_ptoposupcon_queue2048",
    "dinov2_giant_reg_patch_mgn_cov_ptoposupcon_queue2048",
    "dinov2_giant_reg_patch_mgn_cov_suffix8_ptoposupcon_queue2048",
    "radio_v25_b_patch_mgn_cov_ptoposupcon_queue2048",
    "bioclip2_vitl14_projected_patch_mgn_cov_ptoposupcon_queue2048",
    "bioclip_vitb_patch_mgn_cov_ptoposupcon_queue2048",
    "dinov3_huge_plus_patch_mgn_cov_adapt_cltp_jpm4",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_decisionalign",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcltp",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcanon",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headgrid10",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headqvlora8",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_sourcepairlogitkd",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fulladapt32",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxbase",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxext",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_extheadrep24tail2",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headtailother",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_ptransport32",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_head2vpbsupcon",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_fgaux",
    "dinov3_huge_plus_patch_mgn_cov_adapt_qvlora8",
    "dinov3_huge_plus_patch_mgn_cov_adapt_partqvlora4x4",
    "dinov3_convnext_large_dual_mgn_cov",
    "dinov3_convnext_large_dual_mgn_cov_ptoposupcon_queue2048",
    "dinov3_convnext_large_dual_mgn_cov_part_adapter",
    "petface_r50_head_specialist",
}


@contextmanager
def deterministic_cudnn_evaluation(enabled: bool):
    """Make convolutional validation replayable without changing train mode."""
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    if enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EXTERNAL_REID_CLASS_STATE_KEYS = {
    "shared_class_weight",
    "part_class_delta",
    "part_class_available",
}
EXTERNAL_REID_FRESH_QUEUE_PREFIX = "instance_queue_"
EXTERNAL_HEAD_EXPERT_SOURCE_PREFIXES = (
    "branch_projections.",
    "branch_necks.",
)


def load_external_reid_representation(
    model: torch.nn.Module,
    checkpoint_path: Path,
    audit_path: Path,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Strictly transfer external representation tensors, never ID heads."""
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise AssertionError(
            f"External ReID checkpoint SHA-256 {checkpoint_sha256} != "
            f"{expected_checkpoint_sha256}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required_audit = {
        "status": "PASS",
        "kind": "external_animal_identity_pretraining",
        "dataset": "DogFaceNet_224resized",
        "license": "CC-BY-4.0",
        "hyena_or_hyenaid_present": False,
        "competition_data_used": False,
        "competition_checkpoint_loaded": False,
        "model_variant": (
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon"
        ),
        "epochs": 16,
        "fixed_final_epoch_export": True,
        "checkpoint_sha256": checkpoint_sha256,
    }
    for key, expected in required_audit.items():
        if audit.get(key) != expected:
            raise AssertionError(
                f"External ReID audit {key}={audit.get(key)!r} != {expected!r}"
            )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key, expected in {
        "kind": "external_animal_identity_pretraining",
        "dataset": "DogFaceNet_224resized",
        "license": "CC-BY-4.0",
        "hyena_or_hyenaid_present": False,
        "competition_data_used": False,
        "competition_checkpoint_loaded": False,
        "epochs": 16,
        "fixed_final_epoch_export": True,
    }.items():
        if payload.get(key) != expected:
            raise AssertionError(
                f"External ReID payload {key}={payload.get(key)!r} != {expected!r}"
            )
    if payload.get("dataset_manifest_sha256") != audit.get(
        "dataset_manifest_sha256"
    ) or payload.get("dataset_audit_sha256") != audit.get("dataset_audit_sha256"):
        raise AssertionError("External ReID dataset provenance differs from its audit")
    external_state = payload.get("model")
    if not isinstance(external_state, dict):
        raise AssertionError("External ReID checkpoint has no model state")
    target_state = model.state_dict()
    external_class_keys = set(external_state).intersection(
        EXTERNAL_REID_CLASS_STATE_KEYS
    )
    target_class_keys = set(target_state).intersection(EXTERNAL_REID_CLASS_STATE_KEYS)
    if external_class_keys != EXTERNAL_REID_CLASS_STATE_KEYS:
        raise AssertionError(
            f"External ReID class-state keys changed: {external_class_keys}"
        )
    if target_class_keys != EXTERNAL_REID_CLASS_STATE_KEYS:
        raise AssertionError(
            f"Competition class-state keys changed: {target_class_keys}"
        )
    transferable = {
        key: value
        for key, value in external_state.items()
        if key not in EXTERNAL_REID_CLASS_STATE_KEYS
    }
    fresh_queue_keys = {
        key
        for key in target_state
        if key.startswith(EXTERNAL_REID_FRESH_QUEUE_PREFIX)
    }
    if fresh_queue_keys and not getattr(model, "training_instance_queue", False):
        raise AssertionError("External ReID target exposes an undeclared queue")
    if any(key.startswith(EXTERNAL_REID_FRESH_QUEUE_PREFIX) for key in external_state):
        raise AssertionError("External ReID source unexpectedly contains target queue state")
    expected_transferable = (
        set(target_state) - EXTERNAL_REID_CLASS_STATE_KEYS - fresh_queue_keys
    )
    if set(transferable) != expected_transferable:
        missing = sorted(expected_transferable - set(transferable))
        unexpected = sorted(set(transferable) - expected_transferable)
        raise AssertionError(
            f"External representation key mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    shape_mismatches = {
        key: (tuple(value.shape), tuple(target_state[key].shape))
        for key, value in transferable.items()
        if tuple(value.shape) != tuple(target_state[key].shape)
    }
    if shape_mismatches:
        raise AssertionError(
            f"External representation shape mismatch: {shape_mismatches}"
        )
    external_class_shapes = {
        key: list(external_state[key].shape)
        for key in sorted(EXTERNAL_REID_CLASS_STATE_KEYS)
    }
    fresh_target_class_shapes = {
        key: list(target_state[key].shape)
        for key in sorted(EXTERNAL_REID_CLASS_STATE_KEYS)
    }
    result = model.load_state_dict(transferable, strict=False)
    expected_fresh = EXTERNAL_REID_CLASS_STATE_KEYS | fresh_queue_keys
    if set(result.missing_keys) != expected_fresh:
        raise AssertionError(
            f"Fresh competition class keys changed: {result.missing_keys}"
        )
    if result.unexpected_keys:
        raise AssertionError(
            f"Unexpected external representation keys: {result.unexpected_keys}"
        )
    del payload, external_state, transferable
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "dataset": audit["dataset"],
        "dataset_manifest_sha256": audit["dataset_manifest_sha256"],
        "dataset_audit_sha256": audit["dataset_audit_sha256"],
        "loaded_representation_tensors": len(expected_transferable),
        "discarded_external_and_fresh_target_class_keys": sorted(
            EXTERNAL_REID_CLASS_STATE_KEYS
        ),
        "discarded_external_class_shapes": external_class_shapes,
        "fresh_target_class_shapes": fresh_target_class_shapes,
        "fresh_target_queue_keys": sorted(fresh_queue_keys),
        "competition_data_used_in_pretraining": False,
        "competition_checkpoint_loaded_in_pretraining": False,
    }


def _state_subset_sha256(
    state: dict[str, torch.Tensor], keys: set[str]
) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_external_head_expert(
    model: torch.nn.Module,
    checkpoint_path: Path,
    audit_path: Path,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Transfer only external projection/BN state into the head auxiliary."""
    if not getattr(model, "head_identity_expert", False):
        raise AssertionError("External head transfer requires a head expert")
    if getattr(model, "head_identity_expert_detach", True):
        raise AssertionError("V7.2 head auxiliary must propagate descriptor gradients")
    if getattr(model, "head_identity_expert_routing_enabled", True):
        raise AssertionError("V7.2 external expert may not route validation scores")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise AssertionError(
            f"External head checkpoint SHA-256 {checkpoint_sha256} != "
            f"{expected_checkpoint_sha256}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "PASS",
        "kind": "external_animal_identity_pretraining",
        "dataset": "DogFaceNet_224resized",
        "license": "CC-BY-4.0",
        "hyena_or_hyenaid_present": False,
        "competition_data_used": False,
        "competition_checkpoint_loaded": False,
        "model_variant": (
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon"
        ),
        "epochs": 16,
        "fixed_final_epoch_export": True,
        "checkpoint_sha256": checkpoint_sha256,
    }.items():
        if audit.get(key) != expected:
            raise AssertionError(
                f"External head audit {key}={audit.get(key)!r} != {expected!r}"
            )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in (
        "kind",
        "dataset",
        "license",
        "hyena_or_hyenaid_present",
        "competition_data_used",
        "competition_checkpoint_loaded",
        "epochs",
        "fixed_final_epoch_export",
    ):
        if payload.get(key) != audit.get(key):
            raise AssertionError(f"External head payload {key} differs from audit")
    external_state = payload.get("model")
    if not isinstance(external_state, dict):
        raise AssertionError("External head checkpoint has no model state")
    target_state = model.state_dict()
    transfer: dict[str, tuple[str, torch.Tensor]] = {}
    for source_key, value in external_state.items():
        if source_key.startswith("branch_projections."):
            target_key = source_key.replace(
                "branch_projections.",
                "head_identity_expert_projections.",
                1,
            )
        elif source_key.startswith("branch_necks."):
            target_key = source_key.replace(
                "branch_necks.", "head_identity_expert_necks.", 1
            )
        else:
            continue
        if target_key not in target_state:
            raise AssertionError(f"Missing V7.2 target tensor {target_key}")
        if tuple(value.shape) != tuple(target_state[target_key].shape):
            raise AssertionError(f"V7.2 head tensor shape changed: {source_key}")
        transfer[target_key] = (source_key, value)
    transferred_values = sum(value.numel() for _, value in transfer.values())
    if len(transfer) != 49 or transferred_values != 6_702_599:
        raise AssertionError(
            f"V7.2 external head allowlist changed: "
            f"{len(transfer)} tensors/{transferred_values} values"
        )
    allowed_targets = set(transfer)
    untouched_targets = set(target_state) - allowed_targets
    untouched_before = _state_subset_sha256(target_state, untouched_targets)
    with torch.no_grad():
        for target_key, (_, value) in transfer.items():
            target_state[target_key].copy_(value)
    untouched_after = _state_subset_sha256(model.state_dict(), untouched_targets)
    if untouched_after != untouched_before:
        raise AssertionError("External head transfer changed a base/target-class tensor")
    for target_key, (_, value) in transfer.items():
        if not torch.equal(model.state_dict()[target_key].cpu(), value.cpu()):
            raise AssertionError(f"V7.2 external tensor copy failed: {target_key}")
    source_prefixes = sorted(
        prefix
        for prefix in EXTERNAL_HEAD_EXPERT_SOURCE_PREFIXES
        if any(key.startswith(prefix) for key in external_state)
    )
    del payload, external_state, transfer
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "dataset": audit["dataset"],
        "dataset_manifest_sha256": audit["dataset_manifest_sha256"],
        "dataset_audit_sha256": audit["dataset_audit_sha256"],
        "source_prefixes": source_prefixes,
        "target_prefixes": [
            "head_identity_expert_projections.",
            "head_identity_expert_necks.",
        ],
        "loaded_tensors": len(allowed_targets),
        "loaded_values": transferred_values,
        "untouched_state_sha256": untouched_after,
        "external_backbone_loaded": False,
        "external_class_state_loaded": False,
        "competition_data_used_in_pretraining": False,
        "competition_checkpoint_loaded_in_pretraining": False,
    }


def load_external_head_representation(
    model: torch.nn.Module,
    checkpoint_path: Path,
    audit_path: Path,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Load only the audited DogFace adapters, final two blocks and norm."""
    if (
        not getattr(model, "external_head_representation", False)
        or not getattr(model, "head_identity_expert", False)
        or getattr(model, "head_identity_expert_detach", True)
        or not getattr(model, "head_identity_expert_routing_enabled", False)
    ):
        raise AssertionError("External head representation model controls changed")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise AssertionError("External head representation checkpoint SHA-256 changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "PASS",
        "kind": "external_animal_identity_pretraining",
        "dataset": "DogFaceNet_224resized",
        "license": "CC-BY-4.0",
        "hyena_or_hyenaid_present": False,
        "competition_data_used": False,
        "competition_checkpoint_loaded": False,
        "model_variant": (
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon"
        ),
        "epochs": 16,
        "fixed_final_epoch_export": True,
        "checkpoint_sha256": checkpoint_sha256,
    }.items():
        if audit.get(key) != expected:
            raise AssertionError(f"External head representation audit {key} changed")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in (
        "kind",
        "dataset",
        "license",
        "hyena_or_hyenaid_present",
        "competition_data_used",
        "competition_checkpoint_loaded",
        "epochs",
        "fixed_final_epoch_export",
    ):
        if payload.get(key) != audit.get(key):
            raise AssertionError(f"External head representation payload {key} changed")
    external_state = payload.get("model")
    if not isinstance(external_state, dict):
        raise AssertionError("External checkpoint has no model state")
    transfer: dict[str, tuple[str, torch.Tensor]] = {}
    for source_key, value in external_state.items():
        target_key: str | None = None
        if (
            source_key.startswith("backbone.blocks.")
            and ".mlp.adapter_" in source_key
            and int(source_key.split(".")[2]) < 24
        ):
            components = source_key.split(".")
            target_key = ".".join(
                ["external_head_prefix_adapters", components[2], *components[4:]]
            )
        elif source_key.startswith("backbone.blocks.30."):
            target_key = source_key.replace(
                "backbone.blocks.30.", "external_head_tail_blocks.0.", 1
            )
        elif source_key.startswith("backbone.blocks.31."):
            target_key = source_key.replace(
                "backbone.blocks.31.", "external_head_tail_blocks.1.", 1
            )
        elif source_key.startswith("backbone.norm."):
            target_key = source_key.replace(
                "backbone.norm.", "external_head_norm.", 1
            )
        if target_key is not None:
            transfer[target_key] = (source_key, value)
    target_state = model.state_dict()
    if len(transfer) != 128 or sum(
        value.numel() for _, value in transfer.values()
    ) != 56_436_736:
        raise AssertionError("External head representation allowlist count changed")
    if any(
        target_key not in target_state
        or tuple(value.shape) != tuple(target_state[target_key].shape)
        for target_key, (_, value) in transfer.items()
    ):
        raise AssertionError("External head representation target geometry changed")
    source_keys = {source_key for source_key, _ in transfer.values()}
    if _state_subset_sha256(external_state, source_keys) != (
        "ee866e7f02befc76ede740cb63c8d3a6383e6641679754b54c18afd753a6089f"
    ):
        raise AssertionError("External head representation byte subset changed")
    untouched_targets = set(target_state) - set(transfer)
    untouched_before = _state_subset_sha256(target_state, untouched_targets)
    with torch.no_grad():
        for target_key, (_, value) in transfer.items():
            target_state[target_key].copy_(value)
    if _state_subset_sha256(model.state_dict(), untouched_targets) != untouched_before:
        raise AssertionError("External head representation touched the generic base")
    del payload, external_state
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "dataset": audit["dataset"],
        "license": audit["license"],
        "loaded_tensors": 128,
        "loaded_values": 56_436_736,
        "loaded_subset_sha256": (
            "ee866e7f02befc76ede740cb63c8d3a6383e6641679754b54c18afd753a6089f"
        ),
        "loaded_groups": ["prefix24_adaptformer", "tail_blocks_30_31", "final_norm"],
        "external_class_state_loaded": False,
        "external_competition_rows": 0,
        "external_competition_checkpoints": 0,
        "external_hyena_or_hyenaid_present": False,
    }


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


@torch.inference_mode()
def build_body_teacher_artifact(
    model: torch.nn.Module,
    train_frame: pd.DataFrame,
    competition_root: Path,
    image_size: int,
    eval_batch_size: int,
    workers: int,
    fold_seed: int,
    augmentation_profile: str,
    fold_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Build equal-view identity centres from fold-train body crops only."""
    body_frame = train_frame.loc[train_frame.part_index.isin((1, 2))].copy()
    if body_frame.empty or body_frame.fold.nunique() > 4:
        raise AssertionError("Body teacher frame is invalid")
    if body_frame.image_path.str.startswith("test/").any():
        raise AssertionError("Anonymous test image found in body teacher frame")
    body_loader = make_eval_loader(
        body_frame,
        competition_root,
        image_size,
        eval_batch_size,
        workers,
        fold_seed,
        augmentation_profile=augmentation_profile,
    )
    features = extract(
        model,
        body_loader,
        device,
        tta_flip=False,
        return_local=False,
    )
    expected_indices = set(body_frame.sample_index.astype(int))
    actual_indices = set(features["sample_index"].tolist())
    if actual_indices != expected_indices:
        raise AssertionError("Body teacher extraction changed the train-only rows")
    embeddings = F.normalize(
        features["shared_embedding"].float().reshape(
            -1, model.branch_count, model.embedding_dim
        ),
        dim=-1,
    )
    feature_labels = features["label_index"].long()
    feature_parts = features["part_index"].long()
    view_sums = torch.zeros(
        2,
        model.num_classes,
        model.branch_count,
        model.embedding_dim,
        dtype=torch.float32,
    )
    view_counts = torch.zeros(2, model.num_classes, dtype=torch.long)
    for view_index, part_index in enumerate((1, 2)):
        mask = feature_parts.eq(part_index)
        view_sums[view_index].index_add_(
            0, feature_labels[mask], embeddings[mask]
        )
        view_counts[view_index] = torch.bincount(
            feature_labels[mask], minlength=model.num_classes
        )
    view_means = torch.zeros_like(view_sums)
    for view_index in range(2):
        available = view_counts[view_index].gt(0)
        view_means[view_index, available] = F.normalize(
            view_sums[view_index, available]
            / view_counts[view_index, available, None, None],
            dim=-1,
        )
    teacher_available = view_counts.gt(0).any(dim=0)
    teacher_prototypes = torch.zeros_like(view_sums[0])
    teacher_prototypes[teacher_available] = F.normalize(
        view_means[:, teacher_available].sum(dim=0), dim=-1
    )
    model.set_body_teacher_prototypes(
        teacher_prototypes, teacher_available
    )
    artifact_path = fold_dir / "body_teacher_prototypes.pt"
    torch.save(
        {
            "prototypes": teacher_prototypes,
            "available": teacher_available,
            "view_counts": view_counts,
            "sample_indices": features["sample_index"].long(),
            "construction": "normalized side means, then equal-view normalized mean",
        },
        artifact_path,
    )
    head_frame = train_frame.loc[train_frame.part_index.eq(0)]
    covered_head_samples = int(
        head_frame.label_index.astype(int).isin(
            torch.where(teacher_available)[0].tolist()
        ).sum()
    )
    return {
        "body_teacher_artifact": str(artifact_path),
        "body_teacher_sha256": sha256_file(artifact_path),
        "body_teacher_samples": int(len(body_frame)),
        "body_teacher_identities": int(teacher_available.sum()),
        "body_teacher_left_identities": int(view_counts[0].gt(0).sum()),
        "body_teacher_right_identities": int(view_counts[1].gt(0).sum()),
        "head_train_samples": int(len(head_frame)),
        "head_train_samples_with_teacher": covered_head_samples,
    }


def run_fold(
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    labels: list[str],
    fold: int,
) -> dict[str, Any]:
    validate_model_augmentation_profile(
        args.model_variant, args.augmentation_profile
    )
    fold_dir = args.output_dir / f"fold_{fold}"
    metrics_path = fold_dir / "metrics.json"
    if metrics_path.is_file() and not args.force:
        print(f"fold={fold} already complete; skipping", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_frame = manifest.loc[manifest.fold != fold].copy()
    valid_frame = manifest.loc[manifest.fold == fold].copy()
    overlap = set(train_frame.source_group).intersection(valid_frame.source_group)
    if overlap:
        raise AssertionError(f"Source leakage in fold {fold}: {len(overlap)}")
    if train_frame.image_path.str.startswith("test/").any():
        raise AssertionError("Anonymous test image found in train manifest")
    edge_partial_side_counts = (
        build_fold_train_edge_side_counts(train_frame, args.competition_root)
        if args.edge_partial_samples > 0
        else None
    )
    geometry_stats = (
        build_crop_geometry_stats(train_frame, args.competition_root)
        if args.model_variant
        in {
            "radio_v25_b_patch_mgn_cov_ptoposupcon_queue2048",
            "tips_l14_hr_patch_mgn_cov_ptoposupcon_queue2048",
            "dinov2_giant_reg_patch_mgn_cov_geosie2_ptoposupcon_queue2048",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_decisionalign",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcltp",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcanon",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headgrid10",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headqvlora8",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_sourcepairlogitkd",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fulladapt32",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxbase",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxext",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_extheadrep24tail2",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headtailother",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_ptransport32",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_head2vpbsupcon",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_fgaux",
        }
        else None
    )
    if geometry_stats is not None and geometry_stats["rows"] != len(train_frame):
        raise AssertionError("Geometry normalization included non-training rows")
    foreground_metadata: dict[str, object] | None = None
    validation_foreground_metadata: dict[str, object] | None = None
    if args.foreground_mode != "none":
        if args.foreground_mask_root is None:
            raise AssertionError("Foreground mode requires a mask artifact")
        _, foreground_metadata = validate_foreground_mask_artifact(
            train_frame,
            args.foreground_mask_root,
            verify_mask_hashes=True,
        )
        if args.foreground_mode in {"sam_view", "sam_part_view"}:
            if args.validation_foreground_mask_root is None:
                raise AssertionError("SAM view requires a validation mask artifact")
            _, validation_foreground_metadata = validate_foreground_mask_artifact(
                valid_frame,
                args.validation_foreground_mask_root,
                verify_mask_hashes=True,
            )
            if (
                foreground_metadata["rows"] != 3253
                or foreground_metadata["valid"] != 2714
                or foreground_metadata["index_sha256"]
                != "29965afc4f158c65e3ed6d44e9e66244c614a8bbc257cdc9a07619a2c26e86ef"
                or validation_foreground_metadata["rows"] != 814
                or validation_foreground_metadata["valid"] != 693
                or validation_foreground_metadata["index_sha256"]
                != "965c31327b8bb6dbe5e4d5d5db6bc9af43744dee2926ae817e3e9bcb215166b1"
            ):
                raise AssertionError("V4 foreground artifacts differ from SPEC")

    fold_seed = args.seed + fold
    set_seed(fold_seed)
    train_loader, train_sampler = make_train_loader(
        train_frame,
        args.competition_root,
        args.image_size,
        args.workers,
        args.identities_per_batch,
        args.images_per_identity,
        fold_seed,
        augmentation_profile=args.augmentation_profile,
        sampler_profile=args.sampler_profile,
        geometry_stats=geometry_stats,
        foreground_mask_root=args.foreground_mask_root,
        foreground_mode=args.foreground_mode,
    )
    valid_loader = make_eval_loader(
        valid_frame,
        args.competition_root,
        args.image_size,
        args.eval_batch_size,
        args.workers,
        fold_seed,
        augmentation_profile=args.augmentation_profile,
        geometry_stats=geometry_stats,
        foreground_mask_root=(
            args.validation_foreground_mask_root
            if args.foreground_mode in {"sam_view", "sam_part_view"}
            else None
        ),
        foreground_mode=(
            args.foreground_mode
            if args.foreground_mode in {"sam_view", "sam_part_view"}
            else "none"
        ),
        return_foreground_mask=(
            args.model_variant
            in {
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcanon",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headgrid10",
            }
        ),
    )
    device = torch.device("cuda")
    edge_partial_side_counts_tensor = (
        torch.as_tensor(
            edge_partial_side_counts,
            dtype=torch.float32,
            device=device,
        )
        if edge_partial_side_counts is not None
        else None
    )
    model = build_model(
        args.model_variant,
        num_classes=len(labels),
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        local_queries=args.local_queries,
        pretrained=True,
        arc_scale=args.arc_scale,
        arc_margin=args.arc_margin,
        part_delta_scale=args.part_delta_scale,
        freeze_blocks=args.freeze_blocks,
        freeze_stages=args.freeze_stages,
        grad_checkpointing=True,
        external_pretrained_path=args.external_pretrained_checkpoint,
    )
    suffix_stochastic_depth_metadata = (
        configure_suffix_stochastic_depth(
            model,
            first_block=args.freeze_blocks,
            drop_probability=args.suffix_drop_path_rate,
        )
        if args.suffix_drop_path_rate > 0.0
        else {"enabled": False}
    )
    external_reid_metadata: dict[str, Any] = {}
    external_head_expert_metadata: dict[str, Any] = {}
    external_reid_checkpoint = getattr(args, "external_reid_checkpoint", None)
    if external_reid_checkpoint is not None:
        if args.init_checkpoint is not None:
            raise AssertionError(
                "External ReID initialization and competition initialization are exclusive"
            )
        external_reid_metadata = load_external_reid_representation(
            model,
            external_reid_checkpoint,
            args.external_reid_audit,
            args.expected_external_reid_sha256,
        )
    external_head_expert_checkpoint = getattr(
        args, "external_head_expert_checkpoint", None
    )
    if external_head_expert_checkpoint is not None:
        if args.init_checkpoint is not None or external_reid_checkpoint is not None:
            raise AssertionError(
                "External head-only transfer excludes all other initialization"
            )
        external_head_expert_metadata = (
            load_external_head_representation(
                model,
                external_head_expert_checkpoint,
                args.external_head_expert_audit,
                args.expected_external_head_expert_sha256,
            )
            if getattr(model, "external_head_representation", False)
            else load_external_head_expert(
                model,
                external_head_expert_checkpoint,
                args.external_head_expert_audit,
                args.expected_external_head_expert_sha256,
            )
        )
    if args.model_variant == "petface_r50_head_specialist" and (
        not model.petface_pretrained_loaded
        or model.petface_backbone_tensor_count != 325
        or model.petface_external_classifier_shape != (175_081, 512)
    ):
        raise AssertionError("PetFace visual-only initialization audit failed")
    initializer_hash: str | None = None
    initializer_missing_keys: list[str] = []
    if args.init_checkpoint is not None:
        initializer_hash = sha256_file(args.init_checkpoint)
        if (
            args.expected_init_sha256 is not None
            and initializer_hash != args.expected_init_sha256
        ):
            raise AssertionError(
                f"Initializer SHA-256 {initializer_hash} != "
                f"expected {args.expected_init_sha256}"
            )
        initializer = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        if initializer.get("labels") != labels:
            raise AssertionError("Initializer label order differs from this fold")
        initializer_config = initializer.get("config", {})
        if int(initializer_config.get("fold", -1)) != fold:
            raise AssertionError("Initializer was selected on a different fold")
        if (
            initializer_config.get("manifest_sha256")
            != EXPECTED_MANIFEST_SHA256
        ):
            raise AssertionError("Initializer used a different fold manifest")
        if args.exact_initializer:
            if initializer_config.get("model_variant") != args.model_variant:
                raise AssertionError(
                    "Exact initializer model variant differs from the target"
                )
            model.load_state_dict(initializer["model"], strict=True)
        else:
            load_result = model.load_state_dict(
                initializer["model"], strict=False
            )
            missing_prefixes = model.stage_initializer_missing_prefixes()
            if not missing_prefixes:
                raise AssertionError("Model declares no initializer extension")
            expected_missing = {
                key
                for key in model.state_dict()
                if key.startswith(missing_prefixes)
            }
            if set(load_result.missing_keys) != expected_missing:
                raise AssertionError(
                    f"Unexpected initializer missing keys: {load_result.missing_keys}"
                )
            if load_result.unexpected_keys:
                raise AssertionError(
                    f"Unexpected initializer keys: {load_result.unexpected_keys}"
                )
            initializer_missing_keys = sorted(load_result.missing_keys)
            if getattr(model, "head_tail_expert_blocks", 0):
                model.initialize_head_tail_experts_from_base()
        del initializer
    if args.texture_stage_only:
        if args.init_checkpoint is None:
            raise AssertionError("Frozen texture stage requires an initializer")
        model.activate_cross_level_texture_stage()
    if args.image_texture_stage_only:
        if args.init_checkpoint is None:
            raise AssertionError("Image-frequency stage requires an initializer")
        model.activate_image_texture_stage()
    if args.body_head_stage_only:
        if args.init_checkpoint is None:
            raise AssertionError("Body-to-head stage requires an initializer")
        model.activate_body_head_stage()
    if args.head_tail_stage_only:
        if args.init_checkpoint is None:
            raise AssertionError("Head-tail expert stage requires an initializer")
        model.activate_head_tail_expert_stage()
    model = model.to(device)
    suffix_startpoint_anchor = (
        capture_suffix_start_point_anchor(model)
        if args.suffix_startpoint_anchor
        else None
    )
    availability = torch.zeros(3, len(labels), dtype=torch.bool)
    availability[
        torch.as_tensor(train_frame.part_index.to_numpy(copy=True)),
        torch.as_tensor(train_frame.label_index.to_numpy(copy=True)),
    ] = True
    model.set_part_availability(availability)
    if hasattr(model, "set_part_counts"):
        part_counts = torch.zeros(3, len(labels), dtype=torch.float32)
        grouped_counts = train_frame.groupby(
            ["part_index", "label_index"]
        ).size()
        for (part_index, label_index), count in grouped_counts.items():
            part_counts[int(part_index), int(label_index)] = float(count)
        model.set_part_counts(part_counts)
    if hasattr(model, "set_class_counts"):
        class_counts = torch.bincount(
            torch.as_tensor(train_frame.label_index.to_numpy(copy=True)),
            minlength=len(labels),
        )
        model.set_class_counts(class_counts)
    teacher_metadata: dict[str, Any] = {}
    if args.body_head_stage_only:
        teacher_metadata = build_body_teacher_artifact(
            model,
            train_frame,
            args.competition_root,
            args.image_size,
            args.eval_batch_size,
            args.workers,
            fold_seed,
            args.augmentation_profile,
            fold_dir,
            device,
        )
    optimizer = build_optimizer(
        model,
        args.backbone_lr,
        args.head_lr,
        args.weight_decay,
        args.layer_decay,
        optimizer_name=args.optimizer,
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=args.epochs * len(train_loader),
        warmup_steps=args.warmup_epochs * len(train_loader),
    )
    amp_initial_scale = (
        1024.0
        if (
            getattr(model, "pattern_a2gc_branch", False)
            or getattr(model, "hierarchical_slot_architecture", False)
            or getattr(model, "identity_query_pooling", False)
            or getattr(model, "continuous_geometry_conditioning", False)
            or getattr(model, "dual_level_convnext", False)
            or getattr(model, "bioclip_backbone", False)
            or getattr(model, "bioclip2_backbone", False)
            or getattr(model, "sam2_hiera_backbone", False)
            or getattr(model, "tips_backbone", False)
            or getattr(model, "model_name", "").startswith(
                "vit_giant_patch14_reg4_dinov2"
            )
        )
        else 65536.0
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=True, init_scale=amp_initial_scale
    )
    model_ema: ModelEMA | None = None
    model_swa: ModelSWA | None = None
    model_ema_decay: float | None = None
    model_ema_warmup_updates = args.warmup_epochs * len(train_loader)
    if args.model_ema_half_life_epochs > 0.0:
        model_ema_decay = math.exp(
            math.log(0.5)
            / (args.model_ema_half_life_epochs * len(train_loader))
        )
        model_ema = ModelEMA(
            model,
            decay=model_ema_decay,
            warmup_updates=model_ema_warmup_updates,
        )
    if args.model_swa_start_epoch > 0:
        model_swa = ModelSWA(model, start_epoch=args.model_swa_start_epoch)
    selection_model = (
        "ema" if model_ema is not None else "swa" if model_swa is not None else "online"
    )

    deterministic_validation = args.model_variant in {
        "sam2_hiera_small_fpn_mgn_cov_ptoposupcon_queue2048",
        "tips_l14_hr_patch_mgn_cov_ptoposupcon_queue2048",
        "dinov2_giant_reg_patch_mgn_cov_geosie2_ptoposupcon_queue2048",
        "dinov2_giant_reg_patch_mgn_cov_ptoposupcon_queue2048",
        "dinov2_giant_reg_patch_mgn_cov_suffix8_ptoposupcon_queue2048",
        "dinov3_convnext_large_dual_mgn_cov",
        "dinov3_convnext_large_dual_mgn_cov_ptoposupcon_queue2048",
        "dinov3_convnext_large_dual_mgn_cov_part_adapter",
    }
    config = serializable_args(args)
    config.update(
        {
            "fold": fold,
            "fold_seed": fold_seed,
            "model_name": model.model_name,
            "pretraining_source": model.pretraining_source,
            "pretraining_sha256": getattr(model, "pretraining_sha256", None),
            "external_pretraining_metadata": getattr(
                model, "external_pretraining_metadata", {}
            ),
            "external_pretrained_sha256": getattr(
                args, "external_pretrained_sha256", None
            ),
            "external_classifier_discarded_shape": getattr(
                model, "petface_external_classifier_shape", None
            ),
            "manifest_sha256": sha256_file(args.manifest),
            "train_samples": len(train_frame),
            "valid_samples": len(valid_frame),
            "torch_version": torch.__version__,
            "amp_initial_scale": amp_initial_scale,
            "init_checkpoint_sha256": initializer_hash,
            "initializer_missing_keys": initializer_missing_keys,
            "external_reid_initialization": external_reid_metadata,
            "external_head_expert_initialization": (
                external_head_expert_metadata
            ),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "training_instance_queue": bool(
                getattr(model, "training_instance_queue", False)
            ),
            "decision_aligned_classification": bool(
                getattr(model, "decision_aligned_classification", False)
            ),
            "instance_queue_capacity": int(
                getattr(model, "instance_queue_capacity", 0)
                if getattr(model, "training_instance_queue", False)
                else 0
            ),
            "instance_queue_initial_size": int(
                getattr(model, "instance_queue_size", torch.zeros((), dtype=torch.long)).item()
                if getattr(model, "training_instance_queue", False)
                else 0
            ),
            "edge_partial_fold_train_side_counts": edge_partial_side_counts,
            "released_crop_geometry_stats": geometry_stats,
            "foreground_mask_artifact": foreground_metadata,
            "validation_foreground_mask_artifact": (
                validation_foreground_metadata
            ),
            "deterministic_validation": deterministic_validation,
            "selection_model": selection_model,
            "suffix_startpoint_anchor": (
                suffix_startpoint_anchor.metadata()
                if suffix_startpoint_anchor is not None
                else {"enabled": False}
            ),
            "suffix_stochastic_depth": suffix_stochastic_depth_metadata,
            "model_ema_decay": model_ema_decay,
            "model_ema_warmup_updates": (
                model_ema_warmup_updates if model_ema is not None else 0
            ),
            "model_swa_start_epoch": (
                model_swa.start_epoch if model_swa is not None else 0
            ),
            **teacher_metadata,
        }
    )
    write_json(config, fold_dir / "config.json")
    log_path = fold_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    best_score = -1.0
    best_epoch = -1
    stale_epochs = 0
    started = time.time()
    if args.init_checkpoint is not None:
        with deterministic_cudnn_evaluation(deterministic_validation):
            initial_metrics, initial_frame = classifier_metrics(
                model,
                valid_loader,
                manifest,
                labels,
                device,
                tta_flip=False,
            )
        initial_score = float(initial_metrics["final_score"])
        if args.expected_initial_score is not None and abs(
            initial_score - args.expected_initial_score
        ) > 1e-10:
            raise AssertionError(
                f"Initializer score {initial_score} != "
                f"expected {args.expected_initial_score}"
            )
        best_score = initial_score
        save_checkpoint(
            {
                "model": model.state_dict(),
                "epoch": -1,
                "labels": labels,
                "config": config,
                "classifier_metrics": initial_metrics,
            },
            fold_dir / "best.pt",
        )
        initial_frame.to_csv(
            fold_dir / "best_val_predictions.csv",
            index=False,
            encoding="utf-8",
        )
        write_json(initial_metrics, fold_dir / "initial_metrics.json")
        print(
            f"fold={fold} INITIAL_SCORE={initial_score:.10f} "
            + " ".join(
                f"{part}={initial_metrics['per_part'][part]['macro_f1']:.4f}"
                for part in PARTS
            ),
            flush=True,
        )
    if args.head_tail_stage_only:
        # The persisted initializer checkpoint keeps the exact mother bypass.
        # Every trained epoch uses the expert route, enabled once before any
        # optimizer update and never controlled by validation feedback.
        model.enable_head_tail_experts()
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
            prototype_weight=args.prototype_weight,
            supcon_temperature=args.supcon_temperature,
            triplet_scale=args.triplet_scale,
            grad_clip=args.grad_clip,
            edge_partial_samples=args.edge_partial_samples,
            edge_partial_min_ratio=args.edge_partial_min_ratio,
            edge_partial_max_ratio=args.edge_partial_max_ratio,
            edge_partial_ce_weight=args.edge_partial_ce_weight,
            edge_partial_consistency_weight=(
                args.edge_partial_consistency_weight
            ),
            edge_partial_side_counts=edge_partial_side_counts_tensor,
            sam_rho=args.sam_rho,
            foreground_aux_weight=args.foreground_aux_weight,
            model_ema=model_ema,
            suffix_head_pcgrad=args.suffix_head_pcgrad,
            suffix_startpoint_anchor=suffix_startpoint_anchor,
            verify_suffix_anchor_correction=False,
            train_patch_mask_ratio=args.train_patch_mask_ratio,
            train_patch_mask_size=args.train_patch_mask_size,
        )
        if model_swa is not None:
            model_swa.update(model, human_epoch=epoch + 1)
        selection_context = (
            model_ema.apply_to(model)
            if model_ema is not None
            else model_swa.apply_to(model)
            if model_swa is not None
            else nullcontext(model)
        )
        with selection_context:
            with deterministic_cudnn_evaluation(deterministic_validation):
                validation, validation_frame = classifier_metrics(
                    model,
                    valid_loader,
                    manifest,
                    labels,
                    device,
                    tta_flip=False,
                )
            score = float(validation["final_score"])
            improved = score > best_score + args.min_delta
            if improved:
                save_checkpoint(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "labels": labels,
                        "config": config,
                        "classifier_metrics": validation,
                    },
                    fold_dir / "best.pt",
                )
                validation_frame.to_csv(
                    fold_dir / "best_val_predictions.csv",
                    index=False,
                    encoding="utf-8",
                )
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
        elif epoch + 1 > args.patience_start_epoch:
            stale_epochs += 1
        else:
            stale_epochs = 0
        record = {
            "epoch": epoch,
            **losses,
            "classifier_metrics": validation,
            "best_classifier_score": best_score,
            "best_epoch": best_epoch,
            "learning_rate_max": max(group["lr"] for group in optimizer.param_groups),
            "seconds": time.time() - epoch_started,
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            "selection_model": selection_model,
            "ema_updates": model_ema.updates if model_ema is not None else 0,
            "ema_decay": model_ema_decay,
            "swa_updates": model_swa.updates if model_swa is not None else 0,
            "swa_snapshots": model_swa.snapshots if model_swa is not None else 0,
            "swa_start_epoch": model_swa.start_epoch if model_swa is not None else 0,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        part_text = " ".join(
            f"{part}={validation['per_part'][part]['macro_f1']:.4f}"
            for part in PARTS
        )
        print(
            f"fold={fold} epoch={epoch + 1:02d}/{args.epochs} "
            f"loss={losses['loss']:.4f} part_ce={losses['part_ce']:.4f} "
            f"shared_ce={losses['shared_ce']:.4f} "
            + (
                f"decision_ce={losses['decision_ce']:.4f} "
                f"classification={losses['classification']:.4f} "
                f"decision_err={losses['decision_classification_decomposition_error']:.1e} "
                if getattr(model, "decision_aligned_classification", False)
                else ""
            )
            + f"score={score:.4f} "
            f"proto_ce={losses['prototype_ce']:.4f} "
            f"aux={losses['auxiliary']:.4f} "
            f"dense={losses['dense_match']:.4f} "
            f"edge_ce={losses['edge_partial_ce']:.4f} "
            f"edge_cons={losses['edge_consistency']:.4f} "
            + (
                f"sam_loss={losses['sam_perturbed_loss']:.4f} "
                f"sam_g1={losses['sam_first_gradient_norm']:.4f} "
                f"sam_g2={losses['sam_second_gradient_norm']:.4f} "
                if args.sam_rho > 0.0
                else ""
            )
            + f"{part_text} best={best_score:.4f}@{best_epoch + 1} "
            + (f"ema_updates={model_ema.updates} " if model_ema is not None else "")
            + (
                f"swa_snapshots={model_swa.snapshots} "
                if model_swa is not None
                else ""
            )
            + (
                f"pcgrad_cos={losses['suffix_pcgrad_cosine']:.4f} "
                f"pcgrad_conflict={losses['suffix_pcgrad_conflict_fraction']:.3f} "
                if args.suffix_head_pcgrad
                else ""
            )
            + (
                f"anchor_norm={losses['suffix_anchor_correction_norm']:.6f} "
                if args.suffix_startpoint_anchor
                else ""
            )
            + f"time={record['seconds']:.1f}s mem={record['gpu_peak_gb']:.1f}GB",
            flush=True,
        )
        if (fold_dir / "STOP").is_file():
            print(f"fold={fold} manual_stop", flush=True)
            break
        if epoch + 1 >= args.min_epochs and stale_epochs >= args.patience:
            print(f"fold={fold} early_stop", flush=True)
            break

    checkpoint = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    with deterministic_cudnn_evaluation(deterministic_validation):
        valid_features = extract(
            model,
            valid_loader,
            device,
            tta_flip=args.tta_flip,
            return_local=args.gallery_decoder,
            local_grid=args.local_grid,
        )
    classifier_frame = prediction_frame(
        manifest, valid_features, valid_features["classifier_score"], labels
    )
    classifier_metrics_final = competition_metrics(classifier_frame)
    decoder = "classifier"
    final_frame = classifier_frame
    final_metrics = classifier_metrics_final
    if args.gallery_decoder:
        deterministic_train_loader = make_eval_loader(
            train_frame,
            args.competition_root,
            args.image_size,
            args.eval_batch_size,
            args.workers,
            fold_seed,
            augmentation_profile=args.augmentation_profile,
        )
        train_features = extract(
            model,
            deterministic_train_loader,
            device,
            tta_flip=args.tta_flip,
            return_local=True,
            local_grid=args.local_grid,
        )
        fused_scores = gallery_fused_scores(
            train_features,
            valid_features,
            len(labels),
            device,
            temperature=args.density_temperature,
            candidate_images=args.candidate_images,
            local_weight=args.local_match_weight,
        )
        final_frame = prediction_frame(
            manifest, valid_features, fused_scores, labels
        )
        final_frame["classifier_predicted_id"] = (
            classifier_frame.predicted_id.to_numpy()
        )
        final_metrics = competition_metrics(final_frame)
        decoder = "classifier_gallery_local"
    final_frame.to_csv(
        fold_dir / "val_predictions.csv", index=False, encoding="utf-8"
    )
    result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "epochs_ran": epoch + 1,
        "elapsed_seconds": time.time() - started,
        "selection_classifier_metrics": checkpoint["classifier_metrics"],
        "decoder": decoder,
        "classifier_metrics": classifier_metrics_final,
        "final_metrics": final_metrics,
        # Compatibility key for older aggregation scripts.
        "fused_metrics": final_metrics,
    }
    write_json(result, metrics_path)
    print(
        f"fold={fold} FINAL_SCORE={final_metrics['final_score']:.4f} "
        + " ".join(
            f"{part}={final_metrics['per_part'][part]['macro_f1']:.4f}"
            for part in PARTS
        ),
        flush=True,
    )
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/v2_0")
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--model-variant",
        choices=(
            "dinov3_vit",
            "dinov3_convnext_dolg",
            "dinov3_convnext_large_dual_mgn_cov",
            "dinov3_convnext_large_dual_mgn_cov_ptoposupcon_queue2048",
            "dinov3_convnext_large_dual_mgn_cov_part_adapter",
            "sam2_hiera_small_fpn_mgn_cov_ptoposupcon_queue2048",
            "tips_l14_hr_patch_mgn_cov_ptoposupcon_queue2048",
            "efficientnetv2_m_subcenter",
            "petface_r50_head_specialist",
            "arbase_mgn",
            "dinov3_patch_mgn",
            "dinov3_patch_bim",
            "dinov3_patch_mgn_sc",
            "dinov3_large_patch_mgn",
            "dinov3_large_patch_mgn_sc",
            "dinov3_large_patch_mgn_pa",
            "dinov3_large_patch_mgn_ada",
            "dinov3_large_patch_mgn_gate",
            "dinov3_large_patch_mgn_sie",
            "lingbot_large_patch_mgn",
            "dinov3_large_patch_mgn_cov",
            "dinov3_huge_plus_patch_mgn_cov",
            "dinov3_huge_plus_patch_mgn_cov_adapt",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_decisionalign",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcltp",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcanon",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headgrid10",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headqvlora8",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_sourcepairlogitkd",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fulladapt32",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxbase",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxext",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_extheadrep24tail2",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headtailother",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_ptransport32",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_head2vpbsupcon",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_fgaux",
            "dinov3_huge_plus_patch_mgn_cov_convpass",
            "dinov3_huge_plus_patch_mgn_cov_adapt_headconv",
            "dinov3_huge_plus_patch_mgn_cov_adapt_partmoe",
            "dinov3_huge_plus_patch_mgn_cov_adapt_a2gc",
            "dinov3_huge_plus_slots_cov_adapt",
            "dinov3_huge_plus_patch_mgn_cov_adapt_dmatch",
            "dinov3_huge_plus_patch_mgn_cov_adapt_pqmil",
            "dinov3_huge_plus_patch_mgn_cov_adapt_cltp",
            "dinov3_huge_plus_patch_mgn_cov_adapt_cltp_jpm4",
            "dinov3_huge_plus_patch_mgn_cov_adapt_qvlora8",
            "dinov3_huge_plus_patch_mgn_cov_adapt_partqvlora4x4",
            "dinov3_huge_plus_patch_mgn_cov_adapt_prcltp",
            "dinov3_huge_plus_patch_mgn_cov_adapt_prcltp_headtail2",
            "dinov3_huge_plus_patch_mgn_cov_adapt_prcltp_b2hproto",
            "dinov3_huge_plus_patch_mgn_cov_adapt_prcltp_imgfreq",
            "dinov3_large_patch_mgn_cov_proto",
            "eva02_large_patch_mgn_cov",
            "eva02_large_patch_mgn_cov_proto",
            "convnextv2_large_mgn_cov",
            "swinv2_large_mgn_cov",
            "swinv2_base_mgn_cov",
            "siglip2_large_patch_mgn_cov",
            "bioclip_vitb_patch_mgn_cov_ptoposupcon_queue2048",
            "bioclip2_vitl14_projected_patch_mgn_cov_ptoposupcon_queue2048",
            "radio_v25_b_patch_mgn_cov_ptoposupcon_queue2048",
            "dinov2_large_reg_patch_mgn_cov",
            "dinov2_giant_reg_patch_mgn_cov_geosie2_ptoposupcon_queue2048",
            "dinov2_giant_reg_patch_mgn_cov_ptoposupcon_queue2048",
            "dinov2_giant_reg_patch_mgn_cov_suffix8_ptoposupcon_queue2048",
            "dinov3_large_patch_mgn_cov_ldam",
            "dinov3_large_patch_mgn_cov_headldam",
            "dinov3_large_patch_mgn_mpncov",
            "dinov3_large_patch_mgn_cov_sc",
            "dinov3_large_patch_mgn_corr",
            "dinov3_large_patch_mgn_cov_grad",
            "dinov3_large_patch_mgn_cov_xlayer",
            "dinov3_large_patch_mgn_orthogonal",
            "dinov3_large_patch_mgn_cov_dsbn",
            "dinov3_large_patch_mgn_cov_mbn",
            "dinov3_large_patch_mgn_cov_mixstyle",
            "dinov3_large_patch_mgn_cov_axis",
            "dinov3_large_patch_mgn_cov_grad_axis",
            "dinov3_large_patch_mgn_cov_grad_topk",
            "dinov3_large_patch_mgn_cov_grad_simpool",
            "dinov3_large_patch_mgn_cov_simpool",
            "dinov3_large_patch_bim",
            "dinov3_patch_hier",
            "dinov3_large_patch_hier",
        ),
        default="dinov3_vit",
    )
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--external-pretrained-checkpoint", type=Path)
    parser.add_argument("--expected-external-pretrained-sha256")
    parser.add_argument("--external-reid-checkpoint", type=Path)
    parser.add_argument("--external-reid-audit", type=Path)
    parser.add_argument("--expected-external-reid-sha256")
    parser.add_argument("--external-head-expert-checkpoint", type=Path)
    parser.add_argument("--external-head-expert-audit", type=Path)
    parser.add_argument("--expected-external-head-expert-sha256")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--expected-init-sha256")
    parser.add_argument("--exact-initializer", action="store_true")
    parser.add_argument("--texture-stage-only", action="store_true")
    parser.add_argument("--image-texture-stage-only", action="store_true")
    parser.add_argument("--body-head-stage-only", action="store_true")
    parser.add_argument("--head-tail-stage-only", action="store_true")
    parser.add_argument("--expected-initial-score", type=float)
    parser.add_argument(
        "--augmentation-profile",
        choices=(
            "default",
            "arbase",
            "arbase_clip",
            "arbase_siglip",
            "arbase_source_jitter",
            "arbase_source_jitter_quality",
            "arbase_source_jitter_quality_body",
            "arbase_letterbox",
            "arbase_head",
            "degraded",
            "arbase_source_jitter_radio",
            "arbase_source_jitter_tips",
        ),
        default="default",
    )
    parser.add_argument(
        "--sampler-profile",
        choices=(
            "cross_part",
            "same_part",
            "balanced_cross_part",
            "source_paired_cross_part",
            "head_primary_cross_part",
            "body_primary_cross_part",
        ),
        default="cross_part",
    )
    parser.add_argument("--local-queries", type=int, default=4)
    parser.add_argument("--local-grid", type=int, default=6)
    parser.add_argument("--identities-per-batch", type=int, default=4)
    parser.add_argument("--images-per-identity", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--min-epochs", type=int, default=13)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--patience-start-epoch", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--freeze-blocks", type=int, default=6)
    parser.add_argument("--freeze-stages", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1.2e-5)
    parser.add_argument("--head-lr", type=float, default=1.5e-4)
    parser.add_argument("--layer-decay", type=float, default=0.82)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--arc-scale", type=float, default=30.0)
    parser.add_argument("--arc-margin", type=float, default=0.20)
    parser.add_argument("--part-delta-scale", type=float, default=0.20)
    parser.add_argument("--shared-ce-weight", type=float, default=0.50)
    parser.add_argument("--supcon-weight", type=float, default=0.15)
    parser.add_argument("--triplet-weight", type=float, default=0.30)
    parser.add_argument("--prototype-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.10)
    parser.add_argument("--triplet-scale", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sam-rho", type=float, default=0.0)
    parser.add_argument("--suffix-head-pcgrad", action="store_true")
    parser.add_argument("--suffix-startpoint-anchor", action="store_true")
    parser.add_argument("--suffix-drop-path-rate", type=float, default=0.0)
    parser.add_argument("--train-patch-mask-ratio", type=float, default=0.0)
    parser.add_argument("--train-patch-mask-size", type=int, default=16)
    parser.add_argument("--model-ema-half-life-epochs", type=float, default=0.0)
    parser.add_argument("--model-swa-start-epoch", type=int, default=0)
    parser.add_argument("--foreground-mask-root", type=Path)
    parser.add_argument("--validation-foreground-mask-root", type=Path)
    parser.add_argument(
        "--foreground-mode",
        choices=("none", "sam_bg", "sam_aux", "sam_view", "sam_part_view"),
        default="none",
    )
    parser.add_argument("--foreground-aux-weight", type=float, default=0.0)
    parser.add_argument("--edge-partial-samples", type=int, default=0)
    parser.add_argument("--edge-partial-min-ratio", type=float, default=0.0)
    parser.add_argument("--edge-partial-max-ratio", type=float, default=0.0)
    parser.add_argument("--edge-partial-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--edge-partial-consistency-weight", type=float, default=0.0
    )
    parser.add_argument("--density-temperature", type=float, default=0.05)
    parser.add_argument("--candidate-images", type=int, default=40)
    parser.add_argument("--local-match-weight", type=float, default=0.40)
    parser.add_argument("--tta-flip", action="store_true")
    parser.add_argument("--gallery-decoder", action="store_true")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.competition_root = args.competition_root.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.external_pretrained_checkpoint is not None:
        args.external_pretrained_checkpoint = (
            args.external_pretrained_checkpoint.resolve()
        )
        if not args.external_pretrained_checkpoint.is_file():
            raise FileNotFoundError(args.external_pretrained_checkpoint)
    if args.external_reid_checkpoint is not None:
        args.external_reid_checkpoint = args.external_reid_checkpoint.resolve()
        if not args.external_reid_checkpoint.is_file():
            raise FileNotFoundError(args.external_reid_checkpoint)
    if args.external_reid_audit is not None:
        args.external_reid_audit = args.external_reid_audit.resolve()
        if not args.external_reid_audit.is_file():
            raise FileNotFoundError(args.external_reid_audit)
    if args.external_head_expert_checkpoint is not None:
        args.external_head_expert_checkpoint = (
            args.external_head_expert_checkpoint.resolve()
        )
        if not args.external_head_expert_checkpoint.is_file():
            raise FileNotFoundError(args.external_head_expert_checkpoint)
    if args.external_head_expert_audit is not None:
        args.external_head_expert_audit = args.external_head_expert_audit.resolve()
        if not args.external_head_expert_audit.is_file():
            raise FileNotFoundError(args.external_head_expert_audit)
    if args.foreground_mask_root is not None:
        args.foreground_mask_root = args.foreground_mask_root.resolve()
        if not args.foreground_mask_root.is_dir():
            raise FileNotFoundError(args.foreground_mask_root)
    if args.validation_foreground_mask_root is not None:
        args.validation_foreground_mask_root = (
            args.validation_foreground_mask_root.resolve()
        )
        if not args.validation_foreground_mask_root.is_dir():
            raise FileNotFoundError(args.validation_foreground_mask_root)
    if (
        args.model_variant in FROM_GENERIC_ONLY_VARIANTS
        and args.init_checkpoint is not None
    ):
        raise ValueError(
            f"{args.model_variant} must train from its declared generic pretraining"
        )
    if args.init_checkpoint is not None:
        args.init_checkpoint = args.init_checkpoint.resolve()
        if not args.init_checkpoint.is_file():
            raise FileNotFoundError(args.init_checkpoint)
    source_pair_variant = (
        args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_sourcepairlogitkd"
    )
    if source_pair_variant:
        declared_source_pair = (
            args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "source_paired_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 8
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
        )
        if not declared_source_pair:
            raise ValueError("V12.6 constants differ from v12/SPEC.md")
    head_primary_variant = (
        args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048"
        and args.sampler_profile == "head_primary_cross_part"
    )
    if args.sampler_profile == "head_primary_cross_part" and not head_primary_variant:
        raise ValueError("Head-primary sampling is isolated to V12.7")
    if head_primary_variant:
        declared_head_primary = (
            args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 5
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
        )
        if not declared_head_primary:
            raise ValueError("V12.7 constants differ from v12/SPEC.md")
    body_primary_variant = (
        args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048"
        and args.sampler_profile == "body_primary_cross_part"
    )
    if args.sampler_profile == "body_primary_cross_part" and not body_primary_variant:
        raise ValueError("Body-primary sampling is isolated to V12.9")
    if body_primary_variant:
        declared_body_primary = (
            args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 5
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and not args.suffix_head_pcgrad
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
        )
        if not declared_body_primary:
            raise ValueError("V12.9 constants differ from v12/SPEC.md")
    suffix_head_pcgrad_variant = (
        args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048"
        and args.suffix_head_pcgrad
    )
    if args.suffix_head_pcgrad and not suffix_head_pcgrad_variant:
        raise ValueError("Suffix head-protected PCGrad is isolated to V12.8")
    if suffix_head_pcgrad_variant:
        declared_suffix_head_pcgrad = (
            args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 5
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
        )
        if not declared_suffix_head_pcgrad:
            raise ValueError("V12.8 constants differ from v12/SPEC.md")
    suffix_startpoint_anchor_variant = (
        args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048"
        and args.suffix_startpoint_anchor
    )
    if args.suffix_startpoint_anchor and not suffix_startpoint_anchor_variant:
        raise ValueError("Suffix start-point anchoring is isolated to V12.10")
    if suffix_startpoint_anchor_variant:
        declared_suffix_startpoint_anchor = (
            args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 5
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and not args.suffix_head_pcgrad
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
        )
        if not declared_suffix_startpoint_anchor:
            raise ValueError("V12.10 constants differ from v12/SPEC.md")
    suffix_stochastic_depth_variant = (
        args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048"
        and args.suffix_drop_path_rate > 0.0
    )
    if args.suffix_drop_path_rate < 0.0:
        raise ValueError("Suffix stochastic-depth probability cannot be negative")
    if args.suffix_drop_path_rate > 0.0 and not suffix_stochastic_depth_variant:
        raise ValueError("Suffix stochastic depth is isolated to V12.11")
    if suffix_stochastic_depth_variant:
        declared_suffix_stochastic_depth = (
            args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 5
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and abs(args.suffix_drop_path_rate - 0.10) < 1e-12
            and args.sam_rho == 0.0
            and not args.suffix_head_pcgrad
            and not args.suffix_startpoint_anchor
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
        )
        if not declared_suffix_stochastic_depth:
            raise ValueError("V12.11 constants differ from v12/SPEC.md")
    if args.train_patch_mask_ratio < 0.0 or args.train_patch_mask_ratio >= 1.0:
        raise ValueError("Training patch-mask ratio must be in [0, 1)")
    if args.train_patch_mask_size <= 0:
        raise ValueError("Training patch-mask size must be positive")
    training_patch_mask_variant = (
        args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048"
        and args.train_patch_mask_ratio > 0.0
    )
    if args.train_patch_mask_ratio > 0.0 and not training_patch_mask_variant:
        raise ValueError("Training patch masking is isolated to V12.12")
    if training_patch_mask_variant:
        declared_training_patch_mask = (
            args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 5
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and abs(args.train_patch_mask_ratio - 0.10) < 1e-12
            and args.train_patch_mask_size == 16
            and args.suffix_drop_path_rate == 0.0
            and args.sam_rho == 0.0
            and not args.suffix_head_pcgrad
            and not args.suffix_startpoint_anchor
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
        )
        if not declared_training_patch_mask:
            raise ValueError("V12.12 constants differ from v12/SPEC.md")
    petface_enabled = args.model_variant == "petface_r50_head_specialist"
    if petface_enabled:
        if args.external_pretrained_checkpoint is None:
            raise ValueError("V3.0 requires the audited PetFace checkpoint")
        args.external_pretrained_sha256 = sha256_file(
            args.external_pretrained_checkpoint
        )
        if (
            args.expected_external_pretrained_sha256
            != PETFACE_R50_PRETRAINING_SHA256
            or args.external_pretrained_sha256
            != PETFACE_R50_PRETRAINING_SHA256
        ):
            raise AssertionError("V3.0 PetFace checkpoint SHA-256 changed")
        declared_petface = (
            args.init_checkpoint is None
            and args.image_size == 224
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase"
            and args.sampler_profile == "cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 64
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 8
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 3e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.foreground_mode == "none"
            and args.foreground_mask_root is None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
        )
        if not declared_petface:
            raise ValueError("V3.0 PetFace constants differ from v3/SPEC.md")
    elif (
        args.model_variant
        == "sam2_hiera_small_fpn_mgn_cov_ptoposupcon_queue2048"
    ):
        if args.external_pretrained_checkpoint is None:
            raise ValueError("V12.14 requires the exact SAM2.1 Hiera-S checkpoint")
        args.external_pretrained_sha256 = sha256_file(
            args.external_pretrained_checkpoint
        )
        if (
            args.expected_external_pretrained_sha256
            != SAM2_HIERA_SMALL_PRETRAINING_SHA256
            or args.external_pretrained_sha256
            != SAM2_HIERA_SMALL_PRETRAINING_SHA256
        ):
            raise AssertionError("V12.14 SAM2.1 checkpoint SHA-256 changed")
        declared_sam2_hiera = (
            args.init_checkpoint is None
            and args.folds in ([0], [1])
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 64
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 5
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 3
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 3e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.train_patch_mask_ratio == 0.0
            and args.suffix_drop_path_rate == 0.0
            and args.sam_rho == 0.0
            and not args.suffix_head_pcgrad
            and not args.suffix_startpoint_anchor
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "none"
            and args.foreground_mask_root is None
            and args.validation_foreground_mask_root is None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
        )
        if not declared_sam2_hiera:
            raise ValueError("V12.14 constants differ from v12/SPEC.md")
    elif args.model_variant == "radio_v25_b_patch_mgn_cov_ptoposupcon_queue2048":
        if args.external_pretrained_checkpoint is None:
            raise ValueError("V12.1 requires the exact RADIO-v2.5-B checkpoint")
        args.external_pretrained_sha256 = sha256_file(
            args.external_pretrained_checkpoint
        )
        if (
            args.expected_external_pretrained_sha256
            != RADIO_V25_B_PRETRAINING_SHA256
            or args.external_pretrained_sha256
            != RADIO_V25_B_PRETRAINING_SHA256
        ):
            raise AssertionError("V12.1 RADIO checkpoint SHA-256 changed")
        declared_radio = (
            args.init_checkpoint is None
            and args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter_radio"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 64
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 8
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 8
            and abs(args.backbone_lr - 3e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
        )
        if not declared_radio:
            raise ValueError("V12.1 RADIO constants differ from v12/SPEC.md")
    elif args.model_variant == "bioclip_vitb_patch_mgn_cov_ptoposupcon_queue2048":
        if args.external_pretrained_checkpoint is None:
            raise ValueError("V10.4 requires the exact BioCLIP checkpoint")
        args.external_pretrained_sha256 = sha256_file(
            args.external_pretrained_checkpoint
        )
        if (
            args.expected_external_pretrained_sha256
            != BIOCLIP_PRETRAINING_SHA256
            or args.external_pretrained_sha256 != BIOCLIP_PRETRAINING_SHA256
        ):
            raise AssertionError("V10.4 BioCLIP checkpoint SHA-256 changed")
        declared_bioclip = (
            args.init_checkpoint is None
            and args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_clip"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 64
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 8
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 8
            and abs(args.backbone_lr - 3e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
        )
        if not declared_bioclip:
            raise ValueError("BioCLIP constants differ from docs/METHOD.md")
    elif args.model_variant == "tips_l14_hr_patch_mgn_cov_ptoposupcon_queue2048":
        if args.external_pretrained_checkpoint is None:
            raise ValueError("V12.15 requires the exact TIPS L/14-HR checkpoint")
        args.external_pretrained_sha256 = sha256_file(
            args.external_pretrained_checkpoint
        )
        if (
            args.expected_external_pretrained_sha256
            != TIPS_L14_HR_PRETRAINING_SHA256
            or args.external_pretrained_sha256 != TIPS_L14_HR_PRETRAINING_SHA256
        ):
            raise AssertionError("V12.15 TIPS checkpoint SHA-256 changed")
        declared_tips = (
            args.init_checkpoint is None
            and args.folds in ([0], [1])
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter_tips"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 32
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 5
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 18
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.train_patch_mask_ratio == 0.0
            and args.suffix_drop_path_rate == 0.0
            and args.sam_rho == 0.0
            and not args.suffix_head_pcgrad
            and not args.suffix_startpoint_anchor
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "none"
            and args.foreground_mask_root is None
            and args.validation_foreground_mask_root is None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
        )
        if not declared_tips:
            raise ValueError("V12.15 constants differ from v12/SPEC.md")
    elif args.model_variant in {
        "dinov2_giant_reg_patch_mgn_cov_geosie2_ptoposupcon_queue2048",
        "dinov2_giant_reg_patch_mgn_cov_ptoposupcon_queue2048",
        "dinov2_giant_reg_patch_mgn_cov_suffix8_ptoposupcon_queue2048",
    }:
        if args.external_pretrained_checkpoint is None:
            raise ValueError("V12.16/V12.17/V12.18 requires the exact DINOv2-G/14 checkpoint")
        args.external_pretrained_sha256 = sha256_file(
            args.external_pretrained_checkpoint
        )
        if (
            args.expected_external_pretrained_sha256
            != DINOV2_GIANT_REGISTER_PRETRAINING_SHA256
            or args.external_pretrained_sha256
            != DINOV2_GIANT_REGISTER_PRETRAINING_SHA256
        ):
            raise AssertionError("V12.16/V12.17/V12.18 DINOv2-G checkpoint SHA-256 changed")
        capacity_compatible = (
            args.model_variant
            == "dinov2_giant_reg_patch_mgn_cov_ptoposupcon_queue2048"
        )
        suffix8 = (
            args.model_variant
            == "dinov2_giant_reg_patch_mgn_cov_suffix8_ptoposupcon_queue2048"
        )
        declared_dinov2_giant = (
            args.init_checkpoint is None
            and args.folds in ([0], [1])
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == (8 if capacity_compatible else 16)
            and args.images_per_identity == 4
            and args.eval_batch_size == 16
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 5
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == (32 if suffix8 else 36)
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 1.5e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.train_patch_mask_ratio == 0.0
            and args.suffix_drop_path_rate == 0.0
            and args.sam_rho == 0.0
            and not args.suffix_head_pcgrad
            and not args.suffix_startpoint_anchor
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
        )
        if not declared_dinov2_giant:
            raise ValueError("V12.16/V12.17/V12.18 constants differ from v12/SPEC.md")
    elif (
        args.model_variant
        == "bioclip2_vitl14_projected_patch_mgn_cov_ptoposupcon_queue2048"
    ):
        if args.external_pretrained_checkpoint is None:
            raise ValueError("V10.6 requires the exact BioCLIP 2 checkpoint")
        args.external_pretrained_sha256 = sha256_file(
            args.external_pretrained_checkpoint
        )
        if (
            args.expected_external_pretrained_sha256
            != BIOCLIP2_PRETRAINING_SHA256
            or args.external_pretrained_sha256 != BIOCLIP2_PRETRAINING_SHA256
        ):
            raise AssertionError("V10.6 BioCLIP 2 checkpoint SHA-256 changed")
        declared_bioclip2 = (
            args.init_checkpoint is None
            and args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_clip"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 32
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 8
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 16
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.edge_partial_samples == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
        )
        if not declared_bioclip2:
            raise ValueError("BioCLIP 2 constants differ from docs/METHOD.md")
    else:
        args.external_pretrained_sha256 = None
        if (
            args.external_pretrained_checkpoint is not None
            or args.expected_external_pretrained_sha256 is not None
        ):
            raise ValueError(
                "External generic checkpoint is isolated to declared variants"
            )
    frozen_stage_count = sum(
        (
            args.texture_stage_only,
            args.image_texture_stage_only,
            args.body_head_stage_only,
            args.head_tail_stage_only,
        )
    )
    if frozen_stage_count > 1:
        raise ValueError("Only one frozen side stage can train at a time")
    edge_partial_enabled = args.edge_partial_samples > 0
    if edge_partial_enabled:
        if args.model_variant != "dinov3_huge_plus_patch_mgn_cov_adapt":
            raise ValueError("V2.60 edge consistency requires the V2.42 model")
        if args.augmentation_profile != "arbase_source_jitter":
            raise ValueError("V2.60 requires organizer-source jitter")
        if args.init_checkpoint is not None:
            raise ValueError("V2.60 must train from generic pretraining")
        declared = (
            args.edge_partial_samples == 8
            and abs(args.edge_partial_min_ratio - 0.12) < 1e-12
            and abs(args.edge_partial_max_ratio - 0.28) < 1e-12
            and abs(args.edge_partial_ce_weight - 0.25) < 1e-12
            and abs(args.edge_partial_consistency_weight - 0.15) < 1e-12
        )
        if not declared:
            raise ValueError("V2.60 edge-partial constants differ from SPEC")
    elif any(
        abs(value) > 0.0
        for value in (
            args.edge_partial_min_ratio,
            args.edge_partial_max_ratio,
            args.edge_partial_ce_weight,
            args.edge_partial_consistency_weight,
        )
    ):
        raise ValueError("Edge-partial weights require --edge-partial-samples")
    geometry_enabled = (
        args.model_variant
        in {
            "radio_v25_b_patch_mgn_cov_ptoposupcon_queue2048",
            "tips_l14_hr_patch_mgn_cov_ptoposupcon_queue2048",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_decisionalign",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcltp",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcanon",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headgrid10",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headqvlora8",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_sourcepairlogitkd",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fulladapt32",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxbase",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxext",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_extheadrep24tail2",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headtailother",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_ptransport32",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_head2vpbsupcon",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_fgaux",
        }
    )
    decision_align_enabled = (
        args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_decisionalign"
    )
    if decision_align_enabled:
        declared_v14_2 = (
            args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 5
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and not edge_partial_enabled
            and not args.suffix_head_pcgrad
            and not args.suffix_startpoint_anchor
            and args.suffix_drop_path_rate == 0.0
            and args.train_patch_mask_ratio == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and frozen_stage_count == 0
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
        )
        if not declared_v14_2:
            raise ValueError("V14.2 constants differ from v14/SPEC.md")
    quality_view_enabled = (
        args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048"
        and args.augmentation_profile
        in {"arbase_source_jitter_quality", "arbase_source_jitter_quality_body"}
    )
    if geometry_enabled:
        expected_geometry_profile = (
            "arbase_source_jitter_radio"
            if args.model_variant
            == "radio_v25_b_patch_mgn_cov_ptoposupcon_queue2048"
            else (
                "arbase_source_jitter_tips"
                if args.model_variant
                == "tips_l14_hr_patch_mgn_cov_ptoposupcon_queue2048"
                else "arbase_source_jitter"
            )
        )
        if (
            args.augmentation_profile != expected_geometry_profile
            and not quality_view_enabled
        ):
            raise ValueError("Geometry variant requires its organizer-source jitter")
        if args.init_checkpoint is not None:
            raise ValueError("V2.61 must train from generic pretraining")
        if edge_partial_enabled:
            raise ValueError("V2.61 excludes V2.60 edge-partial training")
        expected_freeze_blocks = (
            8
            if args.model_variant
            == "radio_v25_b_patch_mgn_cov_ptoposupcon_queue2048"
            else (
                18
                if args.model_variant
                == "tips_l14_hr_patch_mgn_cov_ptoposupcon_queue2048"
                else (
                    36
                    if args.model_variant
                    == "dinov2_giant_reg_patch_mgn_cov_geosie2_ptoposupcon_queue2048"
                    else (
                        32
                        if args.model_variant
                        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fulladapt32"
                        else 24
                    )
                )
            )
        )
        if (
            args.image_size != 448
            or args.freeze_blocks != expected_freeze_blocks
            or args.embedding_dim != 512
        ):
            raise ValueError("Geometry/token constants differ from SPEC")
    v3_convnext_enabled = args.model_variant in {
        "dinov3_convnext_large_dual_mgn_cov",
        "dinov3_convnext_large_dual_mgn_cov_ptoposupcon_queue2048",
        "dinov3_convnext_large_dual_mgn_cov_part_adapter",
    }
    v12_3_convnext_enabled = (
        args.model_variant
        == "dinov3_convnext_large_dual_mgn_cov_ptoposupcon_queue2048"
    )
    foreground_enabled = args.foreground_mode != "none"
    bioclip_enabled = args.model_variant in {
        "bioclip_vitb_patch_mgn_cov_ptoposupcon_queue2048",
        "bioclip2_vitl14_projected_patch_mgn_cov_ptoposupcon_queue2048",
    }
    radio_enabled = (
        args.model_variant
        == "radio_v25_b_patch_mgn_cov_ptoposupcon_queue2048"
    )
    tips_enabled = (
        args.model_variant
        == "tips_l14_hr_patch_mgn_cov_ptoposupcon_queue2048"
    )
    dinov2_giant_enabled = args.model_variant in {
        "dinov2_giant_reg_patch_mgn_cov_geosie2_ptoposupcon_queue2048",
        "dinov2_giant_reg_patch_mgn_cov_ptoposupcon_queue2048",
        "dinov2_giant_reg_patch_mgn_cov_suffix8_ptoposupcon_queue2048",
    }
    sam2_hiera_enabled = (
        args.model_variant
        == "sam2_hiera_small_fpn_mgn_cov_ptoposupcon_queue2048"
    )
    if (
        radio_enabled
        or tips_enabled
        or dinov2_giant_enabled
        or bioclip_enabled
        or sam2_hiera_enabled
    ):
        # The complete V10.4 foreground and lifecycle contract was validated
        # together with its exact external generic checkpoint above.
        pass
    elif quality_view_enabled:
        declared_v11_quality = (
            args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
            and frozen_stage_count == 0
            and geometry_enabled
            and args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 8
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and not edge_partial_enabled
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
        )
        if not declared_v11_quality:
            raise ValueError("Quality-view constants differ from docs/METHOD.md")
    elif foreground_enabled and v12_3_convnext_enabled:
        declared_v12_3 = (
            args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
            and frozen_stage_count == 0
            and not geometry_enabled
            and args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 64
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 8
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 2
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and not edge_partial_enabled
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
        )
        declared_v12_4 = (
            args.foreground_mode == "sam_bg"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is None
            and args.foreground_aux_weight == 0.0
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.external_reid_checkpoint is None
            and args.external_head_expert_checkpoint is None
            and frozen_stage_count == 0
            and not geometry_enabled
            and args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 64
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 8
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 2
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and not edge_partial_enabled
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed in {20260719, 20260813}
        )
        if not (declared_v12_3 or declared_v12_4):
            raise ValueError("ConvNeXt queue constants differ from V12.3/V12.4 SPEC")
    elif foreground_enabled and v3_convnext_enabled:
        declared_v3_1 = (
            args.foreground_mode == "sam_bg"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is None
            and args.foreground_aux_weight == 0.0
            and args.init_checkpoint is None
            and frozen_stage_count == 0
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 64
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 8
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 2
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and not edge_partial_enabled
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
        )
        if not declared_v3_1:
            raise ValueError("V3.1/V3.2 constants differ from v3/SPEC.md")
    elif args.foreground_mode in {"sam_view", "sam_part_view"}:
        declared_v4 = (
            args.model_variant
            in {
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_decisionalign",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcltp",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcanon",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headgrid10",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headqvlora8",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_sourcepairlogitkd",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fulladapt32",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxbase",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxext",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_extheadrep24tail2",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headtailother",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_ptransport32",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_head2vpbsupcon",
            }
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and frozen_stage_count == 0
            and geometry_enabled
            and args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.augmentation_profile == "arbase_source_jitter"
            and (
                (
                    args.model_variant
                    == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2"
                    and (
                        (
                            args.foreground_mode == "sam_view"
                            and args.sampler_profile == "cross_part"
                        )
                        or (
                            args.foreground_mode == "sam_part_view"
                            and args.sampler_profile
                            in {"cross_part", "balanced_cross_part"}
                        )
                    )
                )
                or (
                    args.model_variant
                    in {
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_decisionalign",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcltp",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcanon",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headgrid10",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headqvlora8",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_sourcepairlogitkd",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fulladapt32",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxbase",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxext",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_extheadrep24tail2",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headtailother",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_ptransport32",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_head2vpbsupcon",
                    }
                    and args.foreground_mode == "sam_part_view"
                    and args.sampler_profile
                    == (
                        "head_primary_cross_part"
                        if head_primary_variant
                        else
                        "body_primary_cross_part"
                        if body_primary_variant
                        else
                        "source_paired_cross_part"
                        if args.model_variant
                        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_sourcepairlogitkd"
                        else "balanced_cross_part"
                    )
                )
            )
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience
            == (
                5
                if head_primary_variant
                or body_primary_variant
                or suffix_head_pcgrad_variant
                or suffix_startpoint_anchor_variant
                or suffix_stochastic_depth_variant
                or training_patch_mask_variant
                or args.model_variant
                == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcanon"
                or args.model_variant
                == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headgrid10"
                or args.model_variant
                == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headqvlora8"
                or args.model_variant
                == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcltp"
                or decision_align_enabled
                else 8
            )
            and args.patience_start_epoch
            == (32 if args.model_swa_start_epoch == 32 else 13)
            and args.warmup_epochs == 3
            and args.freeze_blocks
            == (
                32
                if args.model_variant
                == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fulladapt32"
                else 24
            )
            and args.freeze_stages == 2
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and not edge_partial_enabled
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed
            in (
                {20260719, 20260813}
                if args.model_variant
                == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048"
                else {20260719}
            )
        )
        if not declared_v4:
            raise ValueError("V4/V4.1/V4.2 constants differ from v4/SPEC.md")
    elif foreground_enabled:
        # The first V2.63 run retains its historical three-stale config.  All
        # new foreground runs use the later user-requested eight-stale guard;
        # the separate output directory preserves which lifecycle produced
        # each checkpoint.
        foreground_patience = 8
        common_foreground = (
            args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is None
            and geometry_enabled
            and args.init_checkpoint is None
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == foreground_patience
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and not edge_partial_enabled
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
        )
        declared_foreground = (
            args.foreground_mode == "sam_bg"
            and args.model_variant
            == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2"
            and args.foreground_aux_weight == 0.0
        ) or (
            args.foreground_mode == "sam_aux"
            and args.model_variant
            == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_fgaux"
            and abs(args.foreground_aux_weight - 0.20) < 1e-12
        )
        if not common_foreground or not declared_foreground:
            raise ValueError("V2.63/V2.64 foreground constants differ from SPEC")
    elif v3_convnext_enabled:
        raise ValueError(
            "V3.1/V3.2 require their declared train-only SAM background mode"
        )
    elif (
        args.foreground_mask_root is not None
        or args.validation_foreground_mask_root is not None
        or args.foreground_aux_weight != 0.0
        or args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_fgaux"
    ):
        raise ValueError("Foreground model/artifact requires a declared mode")
    external_reid_values = (
        args.external_reid_checkpoint,
        args.external_reid_audit,
        args.expected_external_reid_sha256,
    )
    external_reid_enabled = args.external_reid_checkpoint is not None
    if external_reid_enabled:
        if any(value is None for value in external_reid_values):
            raise ValueError(
                "V7.1 requires checkpoint, companion audit and expected SHA-256"
            )
        declared_external_reid_target = (
            args.model_variant
            in {
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048",
            }
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 8
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and not edge_partial_enabled
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
        )
        if not declared_external_reid_target:
            raise ValueError("External ReID target constants differ from docs/METHOD.md")
        if len(args.expected_external_reid_sha256) != 64:
            raise ValueError("V7.1 expected external checkpoint SHA-256 is invalid")
    elif any(value is not None for value in external_reid_values):
        raise ValueError(
            "External ReID audit/SHA arguments require --external-reid-checkpoint"
        )
    external_head_expert_values = (
        args.external_head_expert_checkpoint,
        args.external_head_expert_audit,
        args.expected_external_head_expert_sha256,
    )
    external_head_expert_enabled = (
        args.external_head_expert_checkpoint is not None
    )
    if external_head_expert_enabled:
        if any(value is None for value in external_head_expert_values):
            raise ValueError(
                "V7.2 requires head checkpoint, companion audit and expected SHA-256"
            )
        declared_external_head = (
            args.model_variant
            in {
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxext",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_extheadrep24tail2",
            }
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and not external_reid_enabled
            and args.folds == [0]
            and args.image_size == 448
            and args.embedding_dim == 512
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 8
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and not edge_partial_enabled
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
        )
        if not declared_external_head:
            raise ValueError("External-head constants differ from docs/METHOD.md")
        if len(args.expected_external_head_expert_sha256) != 64:
            raise ValueError("V7.2 expected external checkpoint SHA-256 is invalid")
    elif any(value is not None for value in external_head_expert_values):
        raise ValueError(
            "External head audit/SHA arguments require its checkpoint"
        )

    ema_enabled = args.model_ema_half_life_epochs > 0.0
    swa_enabled = args.model_swa_start_epoch > 0
    if args.model_ema_half_life_epochs < 0.0:
        raise ValueError("Model EMA half-life cannot be negative")
    if args.model_swa_start_epoch < 0:
        raise ValueError("Model SWA start epoch cannot be negative")
    if ema_enabled and swa_enabled:
        raise ValueError("Model EMA and SWA selection are mutually exclusive")
    if ema_enabled and (
        args.model_variant
        != "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon"
        or abs(args.model_ema_half_life_epochs - 8.0) > 1e-12
        or args.foreground_mode != "sam_part_view"
        or args.sampler_profile != "balanced_cross_part"
        or args.init_checkpoint is not None
        or args.external_pretrained_checkpoint is not None
        or args.folds != [0]
        or args.warmup_epochs != 3
        or args.patience != 8
    ):
        raise ValueError("V4.5 EMA constants differ from v4/SPEC.md")
    if swa_enabled and (
        args.model_variant
        != "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon"
        or args.model_swa_start_epoch != 32
        or args.foreground_mode != "sam_part_view"
        or args.sampler_profile != "balanced_cross_part"
        or args.init_checkpoint is not None
        or args.external_pretrained_checkpoint is not None
        or args.folds != [0]
        or args.image_size != 448
        or args.embedding_dim != 512
        or args.identities_per_batch != 16
        or args.images_per_identity != 4
        or args.eval_batch_size != 20
        or args.epochs != 48
        or args.min_epochs != 13
        or args.patience != 8
        or args.patience_start_epoch != 32
        or args.warmup_epochs != 3
        or args.freeze_blocks != 24
        or abs(args.backbone_lr - 2.4e-5) > 1e-15
        or abs(args.head_lr - 3e-4) > 1e-15
        or abs(args.layer_decay - 0.82) > 1e-12
        or abs(args.weight_decay - 0.05) > 1e-12
        or args.optimizer != "adamw"
        or abs(args.arc_scale - 30.0) > 1e-12
        or abs(args.arc_margin - 0.20) > 1e-12
        or abs(args.part_delta_scale - 0.20) > 1e-12
        or abs(args.shared_ce_weight - 0.50) > 1e-12
        or abs(args.supcon_weight - 0.15) > 1e-12
        or abs(args.triplet_weight - 0.30) > 1e-12
        or abs(args.prototype_weight) > 1e-12
        or abs(args.supcon_temperature - 0.10) > 1e-12
        or abs(args.triplet_scale - 0.10) > 1e-12
        or abs(args.label_smoothing - 0.05) > 1e-12
        or abs(args.grad_clip - 1.0) > 1e-12
        or abs(args.min_delta - 1e-4) > 1e-15
        or args.sam_rho != 0.0
        or edge_partial_enabled
        or args.gallery_decoder
        or not args.tta_flip
        or args.seed != 20260719
    ):
        raise ValueError("V4.6 SWA constants differ from v4/SPEC.md")
    sam_enabled = args.sam_rho > 0.0
    if sam_enabled:
        declared_sam = (
            geometry_enabled
            and args.model_variant
            == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2"
            and abs(args.sam_rho - 0.05) < 1e-12
            and args.optimizer == "adamw"
            and args.init_checkpoint is None
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "cross_part"
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.eval_batch_size == 20
            and args.epochs == 48
            and args.min_epochs == 13
            and args.patience == 3
            and args.patience_start_epoch == 13
            and args.warmup_epochs == 3
            and args.freeze_blocks == 24
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and abs(args.layer_decay - 0.82) < 1e-12
            and abs(args.weight_decay - 0.05) < 1e-12
            and abs(args.arc_scale - 30.0) < 1e-12
            and abs(args.arc_margin - 0.20) < 1e-12
            and abs(args.part_delta_scale - 0.20) < 1e-12
            and abs(args.shared_ce_weight - 0.50) < 1e-12
            and abs(args.supcon_weight - 0.15) < 1e-12
            and abs(args.triplet_weight - 0.30) < 1e-12
            and abs(args.prototype_weight) < 1e-12
            and abs(args.supcon_temperature - 0.10) < 1e-12
            and abs(args.triplet_scale - 0.10) < 1e-12
            and abs(args.label_smoothing - 0.05) < 1e-12
            and abs(args.grad_clip - 1.0) < 1e-12
            and abs(args.min_delta - 1e-4) < 1e-15
            and not edge_partial_enabled
            and not args.gallery_decoder
            and args.tta_flip
            and args.seed == 20260719
        )
        if not declared_sam:
            raise ValueError("V2.62 SAM constants differ from SPEC")
    elif args.sam_rho < 0.0:
        raise ValueError("SAM rho cannot be negative")
    if args.patience_start_epoch < 0:
        raise ValueError("Patience start epoch must be non-negative")
    if args.exact_initializer and args.init_checkpoint is None:
        raise ValueError("Exact initializer mode requires --init-checkpoint")
    if args.exact_initializer and frozen_stage_count:
        raise ValueError(
            "Exact joint initialization and frozen side-only stages are exclusive"
        )
    if args.exact_initializer and (
        args.expected_initial_score is None
        or args.expected_init_sha256 is None
    ):
        raise ValueError(
            "Exact initializer mode requires expected score and SHA-256"
        )
    if frozen_stage_count and (
        args.expected_initial_score is None
        or args.expected_init_sha256 is None
    ):
        raise ValueError(
            "Frozen side stage requires expected initializer score and SHA-256"
        )
    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("fold must be in 0..4")
    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise AssertionError(f"Unexpected manifest hash: {manifest_hash}")
    manifest = pd.read_csv(args.manifest)
    if len(manifest) != 4067 or manifest.fold.nunique() != 5:
        raise AssertionError("Unexpected validation manifest")
    labels = (
        manifest[["individual_id", "label_index"]]
        .drop_duplicates()
        .sort_values("label_index")
        .individual_id.astype(str)
        .tolist()
    )
    if len(labels) != 255:
        raise AssertionError(f"Expected 255 labels, got {len(labels)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_fold(args, manifest, labels, fold) for fold in args.folds]
    scores = [float(result["final_metrics"]["final_score"]) for result in results]
    summary = {
        "folds": args.folds,
        "partial_cv": len(args.folds) < 5,
        "mean_final_score": float(np.mean(scores)),
        "std_final_score": float(np.std(scores)),
        "results": results,
    }
    write_json(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    main()
