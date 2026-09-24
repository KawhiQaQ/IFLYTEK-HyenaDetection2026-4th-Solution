from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import (
    build_crop_geometry_stats,
    build_fold_train_edge_side_counts,
    foreground_safe_background_blur,
    make_eval_loader,
    make_train_loader,
    validate_foreground_mask_artifact,
    validate_model_augmentation_profile,
)
from engine import (
    ModelEMA,
    ModelSWA,
    basic_training_objective,
    branch_cosine_consistency,
    build_edge_partial_batch,
    build_optimizer,
    build_scheduler,
    capture_model_rng_state,
    clip_model_gradients,
    disable_batch_norm_running_stats,
    edge_truncated_views,
    model_supcon,
    restore_batch_norm_running_stats,
    restore_model_rng_state,
    restore_sam_parameters,
    sam_first_step,
    sam_second_step,
)
from losses import (
    balanced_foreground_auxiliary,
    source_aware_dense_chamfer_contrastive,
    source_aware_part_batch_hard,
)
from model import (
    CONVNEXT_LARGE_DINOV3_PRETRAINING_SHA256,
    PETFACE_R50_PRETRAINING_SHA256,
    build_model,
)
from train_cv import (
    EXTERNAL_REID_CLASS_STATE_KEYS,
    load_external_reid_representation,
)


EXPECTED_MANIFEST_SHA256 = (
    "0470a4bb4630061ef06e92c50a74cb15ba457e476714294872ec426239a1ee6f"
)
FROM_GENERIC_ONLY_VARIANTS = {
    "dinov3_huge_plus_patch_mgn_cov_adapt_cltp_jpm4",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
    "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--identities-per-batch", type=int, default=2)
    parser.add_argument("--images-per-identity", type=int, default=2)
    parser.add_argument("--freeze-blocks", type=int, default=6)
    parser.add_argument("--external-pretrained-checkpoint", type=Path)
    parser.add_argument("--expected-external-pretrained-sha256")
    parser.add_argument("--external-reid-checkpoint", type=Path)
    parser.add_argument("--external-reid-audit", type=Path)
    parser.add_argument("--expected-external-reid-sha256")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--expected-init-sha256")
    parser.add_argument("--exact-initializer", action="store_true")
    parser.add_argument("--backbone-lr", type=float, default=2.4e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--sam-rho", type=float, default=0.0)
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
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--edge-partial-samples", type=int, default=0)
    parser.add_argument("--edge-partial-min-ratio", type=float, default=0.0)
    parser.add_argument("--edge-partial-max-ratio", type=float, default=0.0)
    parser.add_argument("--edge-partial-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--edge-partial-consistency-weight", type=float, default=0.0
    )
    parser.add_argument("--texture-stage-only", action="store_true")
    parser.add_argument("--image-texture-stage-only", action="store_true")
    parser.add_argument("--body-head-stage-only", action="store_true")
    parser.add_argument("--head-tail-stage-only", action="store_true")
    parser.add_argument(
        "--model-variant",
        choices=(
            "dinov3_vit",
            "dinov3_convnext_dolg",
            "dinov3_convnext_large_dual_mgn_cov",
            "dinov3_convnext_large_dual_mgn_cov_ptoposupcon_queue2048",
            "dinov3_convnext_large_dual_mgn_cov_part_adapter",
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
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert",
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
            "dinov2_large_reg_patch_mgn_cov",
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
    parser.add_argument(
        "--augmentation-profile",
        choices=(
            "default",
            "arbase",
            "arbase_clip",
            "arbase_siglip",
            "arbase_source_jitter",
            "arbase_letterbox",
            "arbase_head",
            "degraded",
        ),
        default="default",
    )
    parser.add_argument(
        "--sampler-profile",
        choices=("cross_part", "same_part", "balanced_cross_part"),
        default="cross_part",
    )
    args = parser.parse_args()
    args.competition_root = args.competition_root.resolve()
    args.manifest = args.manifest.resolve()
    if args.external_pretrained_checkpoint is not None:
        args.external_pretrained_checkpoint = (
            args.external_pretrained_checkpoint.resolve()
        )
    if args.external_reid_checkpoint is not None:
        args.external_reid_checkpoint = args.external_reid_checkpoint.resolve()
        if not args.external_reid_checkpoint.is_file():
            raise FileNotFoundError(args.external_reid_checkpoint)
    if args.external_reid_audit is not None:
        args.external_reid_audit = args.external_reid_audit.resolve()
        if not args.external_reid_audit.is_file():
            raise FileNotFoundError(args.external_reid_audit)
    if args.init_checkpoint is not None:
        args.init_checkpoint = args.init_checkpoint.resolve()
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
    if sum(
        (
            args.texture_stage_only,
            args.image_texture_stage_only,
            args.body_head_stage_only,
            args.head_tail_stage_only,
        )
    ) > 1:
        raise ValueError("Only one frozen side stage can train at a time")
    stage_only = (
        args.texture_stage_only
        or args.image_texture_stage_only
        or args.body_head_stage_only
        or args.head_tail_stage_only
    )
    if args.exact_initializer and args.init_checkpoint is None:
        raise ValueError("Exact initializer smoke requires a checkpoint")
    if args.exact_initializer and stage_only:
        raise ValueError("Exact joint initializer cannot be a side-only smoke")
    if (
        args.model_variant in FROM_GENERIC_ONLY_VARIANTS
        and args.init_checkpoint is not None
    ):
        raise ValueError(
            f"{args.model_variant} must train from its declared generic pretraining"
        )
    external_reid_values = (
        args.external_reid_checkpoint,
        args.external_reid_audit,
        args.expected_external_reid_sha256,
    )
    external_reid_enabled = args.external_reid_checkpoint is not None
    if external_reid_enabled:
        if any(value is None for value in external_reid_values):
            raise ValueError(
                "V7.1 smoke requires checkpoint, companion audit and expected SHA-256"
            )
        declared_v7_1 = (
            args.model_variant
            == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon"
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and not stage_only
            and not args.full_train
            and args.image_size == 448
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.freeze_blocks == 24
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and args.foreground_mode == "sam_part_view"
            and args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_aux_weight == 0.0
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and args.sam_rho == 0.0
            and args.model_ema_half_life_epochs == 0.0
            and args.model_swa_start_epoch == 0
            and args.edge_partial_samples == 0
        )
        if not declared_v7_1 or len(args.expected_external_reid_sha256) != 64:
            raise ValueError("DogFace transfer smoke constants differ from docs/METHOD.md")
    elif any(value is not None for value in external_reid_values):
        raise ValueError(
            "External ReID audit/SHA arguments require --external-reid-checkpoint"
        )
    petface_enabled = args.model_variant == "petface_r50_head_specialist"
    if petface_enabled:
        if (
            args.external_pretrained_checkpoint is None
            or not args.external_pretrained_checkpoint.is_file()
            or args.expected_external_pretrained_sha256
            != PETFACE_R50_PRETRAINING_SHA256
            or sha256_file(args.external_pretrained_checkpoint)
            != PETFACE_R50_PRETRAINING_SHA256
        ):
            raise AssertionError("V3.0 PetFace checkpoint audit failed")
        if (
            args.init_checkpoint is not None
            or args.image_size != 224
            or args.identities_per_batch != 16
            or args.images_per_identity != 4
            or args.freeze_blocks != 0
            or args.augmentation_profile != "arbase"
            or args.sampler_profile != "cross_part"
            or abs(args.backbone_lr - 3e-5) > 1e-15
            or abs(args.head_lr - 3e-4) > 1e-15
        ):
            raise ValueError("V3.0 smoke constants differ from v3/SPEC.md")
    elif (
        args.external_pretrained_checkpoint is not None
        or args.expected_external_pretrained_sha256 is not None
    ):
        raise ValueError("External PetFace checkpoint is isolated to V3.0")
    edge_partial_enabled = args.edge_partial_samples > 0
    if edge_partial_enabled:
        declared = (
            args.model_variant == "dinov3_huge_plus_patch_mgn_cov_adapt"
            and args.augmentation_profile == "arbase_source_jitter"
            and args.init_checkpoint is None
            and args.edge_partial_samples == 8
            and abs(args.edge_partial_min_ratio - 0.12) < 1e-12
            and abs(args.edge_partial_max_ratio - 0.28) < 1e-12
            and abs(args.edge_partial_ce_weight - 0.25) < 1e-12
            and abs(args.edge_partial_consistency_weight - 0.15) < 1e-12
        )
        if not declared:
            raise ValueError("V2.60 edge-partial smoke differs from SPEC")
    geometry_enabled = (
        args.model_variant
        in {
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headtailother",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_ptransport32",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_head2vpbsupcon",
            "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_fgaux",
        }
    )
    v3_convnext_enabled = args.model_variant in {
        "dinov3_convnext_large_dual_mgn_cov",
        "dinov3_convnext_large_dual_mgn_cov_ptoposupcon_queue2048",
        "dinov3_convnext_large_dual_mgn_cov_part_adapter",
    }
    v12_3_convnext_enabled = (
        args.model_variant
        == "dinov3_convnext_large_dual_mgn_cov_ptoposupcon_queue2048"
    )
    v3_2_enabled = (
        args.model_variant
        == "dinov3_convnext_large_dual_mgn_cov_part_adapter"
    )
    if geometry_enabled:
        if (
            args.augmentation_profile != "arbase_source_jitter"
            or args.init_checkpoint is not None
            or edge_partial_enabled
            or args.image_size != 448
            or args.freeze_blocks != 24
        ):
            raise ValueError("V2.61 geometry smoke differs from SPEC")
    foreground_enabled = args.foreground_mode != "none"
    full_v6_1_enabled = (
        args.full_train
        and foreground_enabled
        and args.foreground_mode == "sam_part_view"
        and args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert"
        and args.foreground_mask_root is not None
        and args.validation_foreground_mask_root is None
        and args.sampler_profile == "balanced_cross_part"
        and args.identities_per_batch == 16
        and args.images_per_identity == 4
        and args.freeze_blocks == 24
        and args.init_checkpoint is None
        and args.external_pretrained_checkpoint is None
        and args.augmentation_profile == "arbase_source_jitter"
        and abs(args.backbone_lr - 2.4e-5) < 1e-15
        and abs(args.head_lr - 3e-4) < 1e-15
        and args.sam_rho == 0.0
        and not edge_partial_enabled
        and not stage_only
    )
    if args.full_train and not (full_v6_1_enabled or (
        foreground_enabled
        and args.foreground_mode == "sam_bg"
        and args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2"
    )):
        raise ValueError("Full-data smoke is locked to declared full recipes")
    if foreground_enabled and v12_3_convnext_enabled:
        declared_v12_3 = (
            args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.foreground_mode == "sam_part_view"
            and args.foreground_aux_weight == 0.0
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and not stage_only
            and not args.full_train
            and args.image_size == 448
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.freeze_blocks == 2
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "balanced_cross_part"
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and args.sam_rho == 0.0
            and not edge_partial_enabled
        )
        declared_v12_4 = (
            args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is None
            and args.foreground_mode == "sam_bg"
            and args.foreground_aux_weight == 0.0
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and not stage_only
            and not args.full_train
            and args.image_size == 448
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.freeze_blocks == 2
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "cross_part"
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and args.sam_rho == 0.0
            and not edge_partial_enabled
        )
        if not (declared_v12_3 or declared_v12_4):
            raise ValueError("ConvNeXt queue smoke differs from V12.3/V12.4 SPEC")
    elif foreground_enabled and v3_convnext_enabled:
        declared_v3_1 = (
            args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is None
            and args.foreground_mode == "sam_bg"
            and args.foreground_aux_weight == 0.0
            and args.init_checkpoint is None
            and not stage_only
            and args.image_size == 448
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.freeze_blocks == 2
            and args.augmentation_profile == "arbase_source_jitter"
            and args.sampler_profile == "cross_part"
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and args.sam_rho == 0.0
            and not edge_partial_enabled
        )
        if not declared_v3_1:
            raise ValueError("V3.1/V3.2 smoke constants differ from v3/SPEC.md")
    elif args.foreground_mode in {"sam_view", "sam_part_view"}:
        declared_v4 = full_v6_1_enabled or (
            args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is not None
            and args.model_variant
            in {
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headtailother",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_ptransport32",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken",
                "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_head2vpbsupcon",
            }
            and geometry_enabled
            and args.foreground_aux_weight == 0.0
            and args.init_checkpoint is None
            and args.external_pretrained_checkpoint is None
            and not stage_only
            and not args.full_train
            and args.image_size == 448
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.freeze_blocks == 24
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
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headtailother",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_ptransport32",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken",
                        "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_head2vpbsupcon",
                    }
                    and args.foreground_mode == "sam_part_view"
                    and args.sampler_profile == "balanced_cross_part"
                )
            )
            and abs(args.backbone_lr - 2.4e-5) < 1e-15
            and abs(args.head_lr - 3e-4) < 1e-15
            and args.sam_rho == 0.0
            and not edge_partial_enabled
        )
        if not declared_v4:
            raise ValueError("V4/V4.1/V4.2 smoke constants differ from v4/SPEC.md")
    elif foreground_enabled:
        declared_foreground = (
            args.foreground_mask_root is not None
            and args.validation_foreground_mask_root is None
            and geometry_enabled
            and args.identities_per_batch == 16
            and args.images_per_identity == 4
            and args.sampler_profile == "cross_part"
            and args.sam_rho == 0.0
            and (
                (
                    args.foreground_mode == "sam_bg"
                    and args.model_variant
                    == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2"
                    and args.foreground_aux_weight == 0.0
                )
                or (
                    args.foreground_mode == "sam_aux"
                    and args.model_variant
                    == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_fgaux"
                    and abs(args.foreground_aux_weight - 0.20) < 1e-12
                )
            )
        )
        if not declared_foreground:
            raise ValueError("V2.63/V2.64 foreground smoke differs from SPEC")
    elif v3_convnext_enabled:
        raise ValueError("V3.1/V3.2 smoke require their declared foreground mode")
    elif (
        args.foreground_mask_root is not None
        or args.validation_foreground_mask_root is not None
        or args.foreground_aux_weight != 0.0
        or args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_fgaux"
    ):
        raise ValueError("Foreground smoke requires its declared mode")
    sam_enabled = args.sam_rho > 0.0
    if sam_enabled:
        if (
            not geometry_enabled
            or args.model_variant
            != "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2"
            or abs(args.sam_rho - 0.05) >= 1e-12
            or args.identities_per_batch != 16
            or args.images_per_identity != 4
            or args.sampler_profile != "cross_part"
        ):
            raise ValueError("V2.62 SAM smoke differs from SPEC")
    elif args.sam_rho < 0.0:
        raise ValueError("SAM rho cannot be negative")
    ema_enabled = args.model_ema_half_life_epochs > 0.0
    swa_enabled = args.model_swa_start_epoch > 0
    if args.model_ema_half_life_epochs < 0.0:
        raise ValueError("Model EMA half-life cannot be negative")
    if args.model_swa_start_epoch < 0:
        raise ValueError("Model SWA start epoch cannot be negative")
    if ema_enabled and swa_enabled:
        raise ValueError("Model EMA and SWA smoke are mutually exclusive")
    if ema_enabled and (
        args.model_variant
        != "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon"
        or abs(args.model_ema_half_life_epochs - 8.0) > 1e-12
        or args.foreground_mode != "sam_part_view"
        or args.sampler_profile != "balanced_cross_part"
        or args.init_checkpoint is not None
        or args.external_pretrained_checkpoint is not None
        or args.identities_per_batch != 16
        or args.images_per_identity != 4
        or args.freeze_blocks != 24
        or sam_enabled
    ):
        raise ValueError("V4.5 EMA smoke constants differ from v4/SPEC.md")
    if swa_enabled and (
        args.model_variant
        != "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon"
        or args.model_swa_start_epoch != 32
        or args.foreground_mode != "sam_part_view"
        or args.sampler_profile != "balanced_cross_part"
        or args.init_checkpoint is not None
        or args.external_pretrained_checkpoint is not None
        or args.identities_per_batch != 16
        or args.images_per_identity != 4
        or args.freeze_blocks != 24
        or sam_enabled
    ):
        raise ValueError("V4.6 SWA smoke constants differ from v4/SPEC.md")
    prototype_transport_enabled = (
        args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_ptransport32"
    )
    if prototype_transport_enabled and (
        ema_enabled
        or swa_enabled
        or args.foreground_mode != "sam_part_view"
        or args.sampler_profile != "balanced_cross_part"
        or args.init_checkpoint is not None
        or args.external_pretrained_checkpoint is not None
        or args.identities_per_batch != 16
        or args.images_per_identity != 4
        or args.freeze_blocks != 24
        or sam_enabled
    ):
        raise ValueError(
            "V4.7 prototype-transport smoke constants differ from v4/SPEC.md"
        )
    foreground_token_enabled = (
        args.model_variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken"
    )
    if foreground_token_enabled and (
        ema_enabled
        or swa_enabled
        or args.foreground_mode != "sam_part_view"
        or args.sampler_profile != "balanced_cross_part"
        or args.init_checkpoint is not None
        or args.external_pretrained_checkpoint is not None
        or args.identities_per_batch != 16
        or args.images_per_identity != 4
        or args.freeze_blocks != 24
        or sam_enabled
    ):
        raise ValueError(
            "V4.8 foreground-token smoke constants differ from v4/SPEC.md"
        )
    validate_model_augmentation_profile(
        args.model_variant, args.augmentation_profile
    )
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("Unexpected fixed-fold manifest SHA-256")
    manifest = pd.read_csv(args.manifest)
    if len(manifest) != 4067 or int((manifest.fold == 0).sum()) != 814:
        raise AssertionError("Unexpected fixed-fold manifest geometry")
    if manifest.image_path.astype(str).str.startswith("test/").any():
        raise AssertionError("Anonymous test image found in fixed-fold manifest")
    if args.full_train:
        train = manifest.copy()
        valid = manifest.iloc[0:0].copy()
    else:
        train = manifest.loc[manifest.fold != 0].copy()
        valid = manifest.loc[manifest.fold == 0]
        if set(train.source_group).intersection(valid.source_group):
            raise AssertionError("source leakage")
    foreground_metadata = None
    validation_foreground_metadata = None
    if foreground_enabled:
        foreground_index, foreground_metadata = validate_foreground_mask_artifact(
            train,
            args.foreground_mask_root,
            verify_mask_hashes=True,
        )
        expected_foreground = (
            (4067, 3407, "53fbb173a1ace9a5cdbf8054f7c7b5747e9e30d639940ad13d4f2e5304be8687")
            if args.full_train
            else (3253, 2714, "29965afc4f158c65e3ed6d44e9e66244c614a8bbc257cdc9a07619a2c26e86ef")
        )
        if (
            foreground_metadata["rows"] != expected_foreground[0]
            or foreground_metadata["valid"] != expected_foreground[1]
            or foreground_metadata["index_sha256"] != expected_foreground[2]
        ):
            raise AssertionError("V2.63 foreground artifact changed")
        if (
            args.foreground_mode in {"sam_view", "sam_part_view"}
            and not args.full_train
        ):
            validation_foreground_index, validation_foreground_metadata = (
                validate_foreground_mask_artifact(
                    valid,
                    args.validation_foreground_mask_root,
                    verify_mask_hashes=True,
                )
            )
            if (
                validation_foreground_metadata["rows"] != 814
                or validation_foreground_metadata["valid"] != 693
                or validation_foreground_metadata["index_sha256"]
                != "965c31327b8bb6dbe5e4d5d5db6bc9af43744dee2926ae817e3e9bcb215166b1"
                or len(validation_foreground_index) != len(valid)
            ):
                raise AssertionError("V4 validation foreground artifact changed")
        if args.foreground_mode in {"sam_bg", "sam_view", "sam_part_view"}:
            probe = foreground_index.loc[foreground_index.valid.astype(bool)].iloc[0]
            image = cv2.imread(
                str(args.competition_root / str(probe.image_path)),
                cv2.IMREAD_COLOR,
            )
            mask = cv2.imread(
                str(
                    args.foreground_mask_root
                    / "masks"
                    / f"{int(probe.sample_index):06d}.png"
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            if image is None or mask is None or mask.shape != image.shape[:2]:
                raise AssertionError("Foreground smoke probe geometry is invalid")
            binary = mask > 127
            blur_sigma = (
                0.035
                if args.foreground_mode in {"sam_view", "sam_part_view"}
                else 0.025
            )
            blurred = foreground_safe_background_blur(
                image, binary, blur_sigma
            )
            if (
                not np.array_equal(blurred[binary], image[binary])
                or np.array_equal(blurred[~binary], image[~binary])
            ):
                raise AssertionError("Foreground-safe background blur changed semantics")
    edge_partial_side_counts = (
        build_fold_train_edge_side_counts(train, args.competition_root)
        if edge_partial_enabled
        else None
    )
    geometry_stats = (
        build_crop_geometry_stats(train, args.competition_root)
        if geometry_enabled
        else None
    )
    if geometry_enabled and (
        geometry_stats is None
        or geometry_stats["rows"] != len(train)
        or geometry_stats["feature_names"]
        != ["log_crop_pixels", "log_crop_aspect"]
        or len(geometry_stats["mean"]) != 2
        or len(geometry_stats["std"]) != 2
    ):
        raise AssertionError("Fold-train geometry normalization changed")
    if (
        args.foreground_mode in {"sam_view", "sam_part_view"}
        and not args.full_train
    ):
        validation_probe_frame = valid
        if args.foreground_mode == "sam_part_view":
            valid_masks = validation_foreground_index.loc[
                validation_foreground_index.valid.astype(bool)
            ]
            lead_sample_indices = [
                int(
                    valid_masks.loc[valid_masks.part_index == part_index]
                    .iloc[0]
                    .sample_index
                )
                for part_index in (0, 1, 2)
            ]
            lead_set = set(lead_sample_indices)
            ordered_sample_indices = lead_sample_indices + [
                int(sample_index)
                for sample_index in valid.sample_index
                if int(sample_index) not in lead_set
            ]
            validation_probe_frame = (
                valid.set_index("sample_index")
                .loc[ordered_sample_indices]
                .reset_index()
            )
        validation_loader = make_eval_loader(
            validation_probe_frame,
            args.competition_root,
            args.image_size,
            batch_size=8,
            workers=0,
            seed=20260719,
            augmentation_profile=args.augmentation_profile,
            geometry_stats=geometry_stats,
            foreground_mask_root=args.validation_foreground_mask_root,
            foreground_mode=args.foreground_mode,
            return_foreground_mask=foreground_token_enabled,
        )
        validation_batch_a = next(iter(validation_loader))
        validation_batch_b = next(iter(validation_loader))
        expected_validation_batch_length = (
            7 if foreground_token_enabled else 6 if geometry_enabled else 5
        )
        if (
            len(validation_batch_a) != expected_validation_batch_length
            or len(validation_batch_b) != expected_validation_batch_length
            or validation_batch_a[0].shape != (8, 3, 448, 448)
            or not torch.equal(validation_batch_a[0], validation_batch_b[0])
            or not torch.equal(validation_batch_a[3], validation_batch_b[3])
        ):
            raise AssertionError("V4 validation SAM view is not deterministic")
        if foreground_token_enabled and (
            validation_batch_a[6].shape != (8, 448, 448)
            or not torch.equal(validation_batch_a[6], validation_batch_b[6])
            or not validation_batch_a[6].flatten(1).any(dim=1).any()
        ):
            raise AssertionError("V4.8 validation foreground mask is invalid")
        if args.foreground_mode == "sam_part_view":
            raw_validation_loader = make_eval_loader(
                validation_probe_frame,
                args.competition_root,
                args.image_size,
                batch_size=8,
                workers=0,
                seed=20260719,
                augmentation_profile=args.augmentation_profile,
                geometry_stats=geometry_stats,
            )
            raw_validation_batch = next(iter(raw_validation_loader))
            if (
                len(raw_validation_batch)
                != (6 if geometry_enabled else 5)
                or not torch.equal(
                    validation_batch_a[3], raw_validation_batch[3]
                )
                or not torch.equal(
                    validation_batch_a[2], raw_validation_batch[2]
                )
            ):
                raise AssertionError("V4.1 raw/view validation rows differ")
            batch_parts = validation_batch_a[2]
            batch_sample_indices = validation_batch_a[3]
            head_rows = batch_parts.eq(0)
            validity_lookup = dict(
                zip(
                    validation_foreground_index.sample_index.astype(int),
                    validation_foreground_index.valid.astype(bool),
                    strict=True,
                )
            )
            valid_body_rows = batch_parts.ne(0) & torch.tensor(
                [
                    validity_lookup[int(sample_index)]
                    for sample_index in batch_sample_indices
                ],
                dtype=torch.bool,
            )
            body_row_changed = (
                validation_batch_a[0][valid_body_rows]
                .ne(raw_validation_batch[0][valid_body_rows])
                .flatten(1)
                .any(dim=1)
            )
            if (
                not head_rows.any()
                or not valid_body_rows.any()
                or not torch.equal(
                    validation_batch_a[0][head_rows],
                    raw_validation_batch[0][head_rows],
                )
                or not body_row_changed.all()
            ):
                raise AssertionError("V4.1 validation part observation route changed")
    if edge_partial_enabled and edge_partial_side_counts != [
        [75, 24, 87, 4],
        [26, 32, 157, 127],
        [218, 27, 30, 148],
    ]:
        raise AssertionError("Fold-train edge-side profile changed")
    loader, train_sampler = make_train_loader(
        train,
        args.competition_root,
        args.image_size,
        workers=0,
        identities_per_batch=args.identities_per_batch,
        images_per_identity=args.images_per_identity,
        seed=20260719,
        augmentation_profile=args.augmentation_profile,
        sampler_profile=args.sampler_profile,
        geometry_stats=geometry_stats,
        foreground_mask_root=args.foreground_mask_root,
        foreground_mode=args.foreground_mode,
    )
    if args.sampler_profile == "balanced_cross_part":
        replay_counts = np.zeros(3, dtype=np.int64)
        for sampler_epoch in range(48):
            train_sampler.set_epoch(sampler_epoch)
            for sampled_indices in train_sampler:
                replay_counts += np.bincount(
                    train.iloc[sampled_indices].part_index.to_numpy(
                        dtype=np.int64
                    ),
                    minlength=3,
                )
        if args.full_train:
            if replay_counts.tolist() != [66262, 65214, 65132]:
                raise AssertionError(
                    "Full-data 48-epoch part exposure differs from preregistration"
                )
        else:
            replay_fraction = replay_counts / replay_counts.sum()
            expected_fraction = np.asarray(
                [0.34036714, 0.32993132, 0.32970154], dtype=np.float64
            )
            if not np.allclose(
                replay_fraction, expected_fraction, rtol=0.0, atol=1e-8
            ):
                raise AssertionError(
                    "V4.2 48-epoch part exposure differs from preregistration"
                )
        train_sampler.set_epoch(0)
    model = build_model(
        args.model_variant,
        num_classes=255,
        image_size=args.image_size,
        embedding_dim=512,
        local_queries=4,
        pretrained=args.init_checkpoint is None,
        arc_scale=30.0,
        arc_margin=0.20,
        part_delta_scale=0.20,
        freeze_blocks=args.freeze_blocks,
        freeze_stages=2,
        grad_checkpointing=True,
        external_pretrained_path=args.external_pretrained_checkpoint,
    )
    external_reid_metadata: dict[str, object] = {}
    fresh_external_class_state: dict[str, torch.Tensor] = {}
    if external_reid_enabled:
        fresh_external_class_state = {
            key: model.state_dict()[key].detach().clone()
            for key in EXTERNAL_REID_CLASS_STATE_KEYS
        }
        external_reid_metadata = load_external_reid_representation(
            model,
            args.external_reid_checkpoint,
            args.external_reid_audit,
            args.expected_external_reid_sha256,
        )
        after_external_load = model.state_dict()
        if any(
            not torch.equal(after_external_load[key], fresh_external_class_state[key])
            for key in EXTERNAL_REID_CLASS_STATE_KEYS
        ):
            raise AssertionError("External identity classifier entered V7.1")
        if (
            int(external_reid_metadata["loaded_representation_tensors"]) <= 0
            or external_reid_metadata["discarded_external_class_shapes"]
            == external_reid_metadata["fresh_target_class_shapes"]
        ):
            raise AssertionError("V7.1 representation/classifier transfer audit failed")
    head_expert_enabled = getattr(model, "head_identity_expert", False)
    head_tail_other_enabled = getattr(model, "head_tail_other_expert", False)
    head_expert_width = getattr(
        model, "head_identity_expert_output_classes", 0
    )
    if petface_enabled:
        if (
            not model.petface_pretrained_loaded
            or model.petface_backbone_tensor_count != 325
            or model.petface_external_classifier_shape != (175_081, 512)
            or any(
                tuple(tensor.shape) == (175_081, 512)
                for tensor in model.state_dict().values()
            )
        ):
            raise AssertionError("PetFace classifier was not cleanly discarded")
    if args.init_checkpoint is not None:
        if (
            args.expected_init_sha256 is not None
            and sha256_file(args.init_checkpoint) != args.expected_init_sha256
        ):
            raise AssertionError("Initializer SHA-256 changed")
        checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        if len(checkpoint.get("labels", [])) != 255:
            raise AssertionError("Initializer label space is not the official 255 IDs")
        checkpoint_config = checkpoint.get("config", {})
        if args.exact_initializer:
            if checkpoint_config.get("model_variant") != args.model_variant:
                raise AssertionError("Exact initializer model variant changed")
            model.load_state_dict(checkpoint["model"], strict=True)
        else:
            load_result = model.load_state_dict(
                checkpoint["model"], strict=False
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
            if getattr(model, "head_tail_expert_blocks", 0):
                model.initialize_head_tail_experts_from_base()
        del checkpoint
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
        smoke_teacher = F.normalize(
            model.shared_class_weight.detach().permute(1, 0, 2).float(),
            dim=-1,
        )
        model.set_body_teacher_prototypes(
            smoke_teacher,
            torch.ones(255, dtype=torch.bool),
        )
    if args.head_tail_stage_only:
        if args.init_checkpoint is None:
            raise AssertionError("Head-tail expert stage requires an initializer")
        model.activate_head_tail_expert_stage()
        model.enable_head_tail_experts()
    model = model.cuda().train()
    if v12_3_convnext_enabled and (
        not model.part_topology_supcon
        or not model.training_instance_queue
        or model.instance_queue_capacity != 2048
        or tuple(model.instance_queue_embeddings.shape) != (2048, 7 * 512)
        or model.instance_queue_embeddings.dtype != torch.float16
        or model.instance_queue_embeddings.requires_grad
        or int(model.instance_queue_size) != 0
        or int(model.instance_queue_pointer) != 0
    ):
        raise AssertionError("V12.3 empty detached queue contract changed")
    head_expert_before: torch.Tensor | None = None
    if head_expert_enabled:
        expected_mix = 0.10 if head_tail_other_enabled else 0.25
        expected_expert_width = 256 if head_tail_other_enabled else 255
        copied_expert_weight = torch.equal(
            model.shared_class_weight,
            model.head_identity_expert_class_weight[:, :255],
        )
        copied_other_weight = (
            torch.allclose(
                model.head_identity_expert_class_weight[:, 255],
                model.shared_class_weight.mean(dim=1),
                atol=1e-7,
                rtol=1e-6,
            )
            if head_tail_other_enabled
            else True
        )
        if (
            model.branch_count != 7
            or abs(model.head_identity_expert_loss_weight - 0.50) > 1e-12
            or abs(model.head_identity_expert_mix - expected_mix) > 1e-12
            or head_expert_width != expected_expert_width
            or not copied_expert_weight
            or not copied_other_weight
        ):
            raise AssertionError(
                "V6 fixed expert inventory changed: "
                f"branches={model.branch_count} "
                f"loss_weight={model.head_identity_expert_loss_weight} "
                f"mix={model.head_identity_expert_mix} "
                f"width={head_expert_width} copied={copied_expert_weight} "
                f"other={copied_other_weight}"
            )
        for base_module, expert_module in zip(
            model.branch_projections,
            model.head_identity_expert_projections,
        ):
            if any(
                not torch.equal(base_module.state_dict()[name], value)
                for name, value in expert_module.state_dict().items()
            ):
                raise AssertionError("V6.1 expert projection is not an exact copy")
        for base_module, expert_module in zip(
            model.branch_necks,
            model.head_identity_expert_necks,
        ):
            if any(
                not torch.equal(base_module.state_dict()[name], value)
                for name, value in expert_module.state_dict().items()
            ):
                raise AssertionError("V6.1 expert BN is not an exact copy")
        head_expert_before = (
            model.head_identity_expert_class_weight.detach().clone()
        )
    prototype_transport_up_before: torch.Tensor | None = None
    if prototype_transport_enabled:
        down = model.part_prototype_transport_down
        up = model.part_prototype_transport_up
        if (
            model.prototype_transport_rank != 32
            or abs(model.prototype_transport_scale - 0.20) > 1e-12
            or model.branch_count != 7
            or tuple(down.shape) != (3, 7, 32, 512)
            or tuple(up.shape) != (3, 7, 512, 32)
            or torch.count_nonzero(up).item() != 0
            or torch.count_nonzero(down).item() == 0
        ):
            raise AssertionError("V4.7 prototype-transport inventory changed")
        hidden = torch.einsum(
            "pmrd,mcd->pmcr", down, model.shared_class_weight
        )
        transported = torch.einsum("pmdr,pmcr->pmcd", up, hidden)
        if torch.count_nonzero(transported).item() != 0:
            raise AssertionError("V4.7 transport did not preserve initial classifier")
        prototype_transport_up_before = up.detach().clone()
        del hidden, transported
    v3_convnext_update_probe: tuple[torch.nn.Parameter, torch.Tensor] | None = None
    v3_2_adapter_update_probe: tuple[
        torch.nn.Parameter, torch.Tensor
    ] | None = None
    if v3_convnext_enabled:
        expected_projection_inputs = [3072, 768, 768, 768, 768, 768, 4096]
        actual_projection_inputs = [
            int(projection[0].in_features)
            for projection in model.branch_projections
        ]
        frozen_modules = [model.backbone.stem, *model.backbone.stages[:2]]
        trainable_modules = list(model.backbone.stages[2:])
        if (
            not getattr(model, "dual_level_convnext", False)
            or model.pretraining_sha256
            != CONVNEXT_LARGE_DINOV3_PRETRAINING_SHA256
            or model.backbone.default_cfg.get("hf_hub_id")
            != "timm/convnext_large.dinov3_lvd1689m"
            or model.branch_count != 7
            or actual_projection_inputs != expected_projection_inputs
            or model.covariance_reduction[0].in_features != 768
            or model.covariance_reduction[0].out_features != 64
            or any(
                parameter.requires_grad
                for module in frozen_modules
                for parameter in module.parameters()
            )
            or any(
                not parameter.requires_grad
                for module in trainable_modules
                for parameter in module.parameters()
            )
        ):
            raise AssertionError(
                "V3.1/V3.2 architecture/pretraining/freeze contract changed"
            )
        update_parameter = model.backbone.stages[3].blocks[-1].gamma
        v3_convnext_update_probe = (
            update_parameter,
            update_parameter.detach().clone(),
        )
        if v3_2_enabled:
            adapter_parameters = [
                parameter
                for bank in (
                    model.local_part_adapters,
                    model.final_part_adapters,
                )
                for parameter in bank.parameters()
            ]
            if (
                not getattr(model, "part_routed_spatial_adapters", False)
                or model.local_part_adapters.rank != 64
                or model.final_part_adapters.rank != 64
                or len(model.local_part_adapters.routes) != 3
                or len(model.final_part_adapters.routes) != 3
                or sum(parameter.numel() for parameter in adapter_parameters)
                != 888_192
            ):
                raise AssertionError("V3.2 routed spatial adapter inventory changed")
            adapter_parameter = model.local_part_adapters.routes[0][4].weight
            v3_2_adapter_update_probe = (
                adapter_parameter,
                adapter_parameter.detach().clone(),
            )
    if getattr(model, "frozen_prefix_adaptformer", False):
        expected_adapters = min(args.freeze_blocks, len(model.backbone.blocks))
        if len(model.prefix_adaptformer_modules) != expected_adapters:
            raise AssertionError(
                f"AdaptFormer module count {len(model.prefix_adaptformer_modules)} "
                f"!= {expected_adapters}"
            )
        if args.init_checkpoint is None and not external_reid_enabled and any(
            torch.count_nonzero(module.adapter_up.weight).item()
            or torch.count_nonzero(module.adapter_up.bias).item()
            for module in model.prefix_adaptformer_modules
        ):
            raise AssertionError("AdaptFormer did not start with zero output")
    if getattr(model, "continuous_geometry_conditioning", False):
        geometry_parameters = model.geometry_conditioning_parameters()
        if (
            model.branch_count != 7
            or len(geometry_parameters) != 4
            or sum(parameter.numel() for parameter in geometry_parameters)
            != 496_128
        ):
            raise AssertionError("V2.61 geometry conditioner inventory changed")
        geometry_output_nonzero = bool(
            torch.count_nonzero(model.geometry_conditioner[-1].weight).item()
            or torch.count_nonzero(model.geometry_conditioner[-1].bias).item()
        )
        if external_reid_enabled != geometry_output_nonzero:
            raise AssertionError(
                "Geometry conditioner zero/nonzero state differs from the declared initializer"
            )
    if getattr(model, "foreground_auxiliary", False):
        foreground_parameters = list(model.foreground_auxiliary_head.parameters())
        if (
            model.branch_count != 7
            or sum(parameter.numel() for parameter in foreground_parameters)
            != 3_841
        ):
            raise AssertionError("V2.64 foreground head inventory changed")
    if getattr(model, "frozen_prefix_qv_lora", False):
        if args.init_checkpoint is not None:
            raise AssertionError("Q/V LoRA smoke illegally loaded a competition checkpoint")
        expected_modules = min(args.freeze_blocks, len(model.backbone.blocks))
        routed_qv = bool(getattr(model, "part_routed_qv_lora", False))
        expected_rank = 4 if routed_qv else 8
        expected_alpha = 4.0 if routed_qv else 8.0
        expected_matrix_count = 384 if routed_qv else 96
        expected_parameter_count = 1_966_080 if routed_qv else 983_040
        if (
            args.freeze_blocks != 24
            or len(model.prefix_qv_lora_modules) != expected_modules
            or model.qv_lora_rank != expected_rank
            or model.qv_lora_part_rank != (4 if routed_qv else 0)
            or model.qv_lora_alpha != expected_alpha
            or model.qv_lora_dropout != 0.05
            or model.branch_count != 7
            or getattr(model, "jpm_local_branches", 0)
            or getattr(model, "cross_level_texture_pyramid", False)
        ):
            raise AssertionError("Q/V LoRA inventory or isolated V2.42 geometry changed")
        if len(model.prefix_qv_lora_parameters()) != expected_matrix_count:
            raise AssertionError(
                f"Q/V LoRA must expose exactly {expected_matrix_count} low-rank matrices"
            )
        if sum(
            parameter.numel() for parameter in model.prefix_qv_lora_parameters()
        ) != expected_parameter_count:
            raise AssertionError("Q/V LoRA parameter count changed")
        for module in model.prefix_qv_lora_modules:
            common_invalid = (
                module.alpha != expected_alpha
                or module.dropout_probability != 0.05
                or module.base_qkv.in_features != 1280
                or module.base_qkv.out_features != 3840
                or torch.count_nonzero(module.query_up.weight).item()
                or torch.count_nonzero(module.value_up.weight).item()
                or not torch.count_nonzero(module.query_down.weight).item()
                or not torch.count_nonzero(module.value_down.weight).item()
            )
            if routed_qv:
                routed_invalid = (
                    module.shared_rank != 4
                    or module.part_rank != 4
                    or module.shared_scale != 1.0
                    or module.part_scale != 1.0
                    or module.num_parts != 3
                    or any(
                        torch.count_nonzero(up.weight).item()
                        for up in (
                            *module.query_part_up,
                            *module.value_part_up,
                        )
                    )
                    or any(
                        not torch.count_nonzero(down.weight).item()
                        for down in (
                            *module.query_part_down,
                            *module.value_part_down,
                        )
                    )
                )
            else:
                routed_invalid = (
                    module.rank != 8
                    or module.scale != 1.0
                )
            if common_invalid or routed_invalid:
                raise AssertionError("Q/V LoRA geometry or exact-zero start changed")
            if not all(
                parameter.requires_grad for parameter in module.adapter_parameters()
            ) or any(
                parameter.requires_grad for parameter in module.base_qkv.parameters()
            ):
                raise AssertionError("Q/V LoRA trainable/frozen boundary changed")
            module.audit_residual = True
    if getattr(model, "prefix_convpass_modules", None):
        multiplier = 1 if getattr(model, "head_convpass_attention", False) else 2
        expected_modules = multiplier * min(
            args.freeze_blocks, len(model.backbone.blocks)
        )
        if len(model.prefix_convpass_modules) != expected_modules:
            raise AssertionError(
                f"ConvPass module count {len(model.prefix_convpass_modules)} "
                f"!= {expected_modules}"
            )
        probe_module = model.prefix_convpass_modules[0]
        probe = torch.randn(
            2,
            probe_module.prefix_tokens + 16,
            probe_module.adapter_down.in_features,
            device="cuda",
        )
        with torch.no_grad():
            probe_delta = probe_module.adapter_forward(probe)
        if not torch.equal(probe_delta, torch.zeros_like(probe_delta)):
            raise AssertionError("zero-initialized ConvPass changed the base function")
        if not all(
            parameter.requires_grad
            for parameter in model.prefix_convpass_parameters()
        ):
            raise AssertionError("ConvPass parameter was accidentally frozen")
        del probe, probe_delta
    if getattr(model, "part_mlp_expert_modules", None):
        if len(model.part_mlp_expert_modules) != 4:
            raise AssertionError("Part MLP-MoE must occupy exactly four tail blocks")
        extra_parameters = sum(
            module.extra_parameter_count()
            for module in model.part_mlp_expert_modules
        )
        if extra_parameters != 157_378_560:
            raise AssertionError(
                f"Unexpected MLP expert parameter count: {extra_parameters}"
            )
        for module in model.part_mlp_expert_modules:
            reference = dict(module.experts[0].named_parameters())
            for expert in module.experts[1:]:
                candidate = dict(expert.named_parameters())
                if candidate.keys() != reference.keys() or any(
                    not torch.equal(reference[name], candidate[name])
                    for name in reference
                ):
                    raise AssertionError("MLP experts did not start identically")
    if getattr(model, "pattern_a2gc_branch", False):
        aggregator = model.pattern_aggregator
        if aggregator is None or model.branch_count != 8:
            raise AssertionError("A2GC did not add exactly one descriptor branch")
        if aggregator.output_dim != 8448:
            raise AssertionError(f"Unexpected A2GC output width: {aggregator.output_dim}")
        aggregator_parameters = sum(
            parameter.numel() for parameter in aggregator.parameters()
        )
        if aggregator_parameters != 2_266_230:
            raise AssertionError(
                f"Unexpected A2GC parameter count: {aggregator_parameters}"
            )
        probe_parts = torch.tensor([0, 1, 2], device="cuda")
        with torch.no_grad():
            geometry = aggregator.geometry_scores(
                probe_parts,
                14,
                14,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )
            mirrored_geometry = aggregator.geometry_scores(
                probe_parts,
                14,
                14,
                dtype=torch.float32,
                device=torch.device("cuda"),
                mirror_x=True,
            )
            if not torch.equal(geometry[1:], mirrored_geometry[1:]):
                raise AssertionError("Body A2GC geometry is not exactly mirror invariant")
            if torch.equal(geometry[0], mirrored_geometry[0]):
                raise AssertionError("Head A2GC geometry unexpectedly lost signed x")
            probe_scores = torch.zeros(
                3, aggregator.num_clusters, 196, device="cuda"
            )
            probe_transport = aggregator.asymmetric_transport(
                probe_scores, aggregator.dustbin_score[probe_parts]
            )
            expected_column_mass = 1.0 / (
                196 + aggregator.num_clusters
            )
            if not torch.isfinite(probe_transport).all() or torch.any(
                probe_transport <= 0
            ):
                raise AssertionError("A2GC transport is non-finite or empty")
            column_mass = probe_transport.sum(dim=1)
            if not torch.allclose(
                column_mass,
                torch.full_like(column_mass, expected_column_mass),
                atol=1e-6,
                rtol=1e-5,
            ):
                raise AssertionError("A2GC target marginal calibration failed")
        del geometry, mirrored_geometry, probe_scores, probe_transport
    if getattr(model, "hierarchical_slot_architecture", False):
        aggregator = model.slot_aggregator
        if aggregator is None or model.branch_count != 7:
            raise AssertionError("Hierarchical slots did not replace the MGN branches")
        if aggregator.num_identity_slots != 4 or aggregator.num_slots != 5:
            raise AssertionError("Hierarchical slot inventory is incorrect")
        if aggregator.iterations != 3 or aggregator.reconstruction_weight != 0.10:
            raise AssertionError("Hierarchical slot constants changed")
        if len(model.branch_projections) != 7 or any(
            projection[0].in_features != 512
            for projection in model.branch_projections[1:6]
        ):
            raise AssertionError("Hierarchical slot descriptor widths changed")
        probe_parts = torch.tensor([0, 1, 2], device="cuda")
        with torch.no_grad():
            coordinates = aggregator.coordinate_features(
                probe_parts,
                14,
                14,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )
            mirrored_coordinates = aggregator.coordinate_features(
                probe_parts,
                14,
                14,
                dtype=torch.float32,
                device=torch.device("cuda"),
                mirror_x=True,
            )
        if not torch.equal(coordinates[1:], mirrored_coordinates[1:]):
            raise AssertionError("Body slot coordinates are not mirror invariant")
        if torch.equal(coordinates[0], mirrored_coordinates[0]):
            raise AssertionError("Head slot coordinates unexpectedly lost signed x")
        del coordinates, mirrored_coordinates
    if getattr(model, "dense_correspondence_training", False):
        if model.branch_count != 7 or len(model.branch_projections) != 7:
            raise AssertionError("Dense correspondence changed inference branches")
        if (
            model.dense_correspondence_topk != 16
            or model.dense_correspondence_temperature != 0.07
            or model.dense_correspondence_weight != 0.20
        ):
            raise AssertionError("Dense correspondence constants changed")
        if (
            model.dense_intermediate_projection[-1].out_features != 64
            or model.dense_final_projection[-1].out_features != 64
        ):
            raise AssertionError("Dense correspondence projection width changed")
    if getattr(model, "identity_query_pooling", False):
        if model.branch_count != 7 or len(model.branch_projections) != 7:
            raise AssertionError("Identity-query pooling changed incumbent descriptors")
        if (
            model.identity_query_dim != 128
            or model.identity_query_temperature != 0.07
        ):
            raise AssertionError("Identity-query constants changed")
        if (
            model.identity_query_patch_projection[-1].in_features != 1280
            or model.identity_query_patch_projection[-1].out_features != 128
            or model.identity_query_weight_projection.in_features != 512
            or model.identity_query_weight_projection.out_features != 128
        ):
            raise AssertionError("Identity-query projection geometry changed")
    if getattr(model, "cross_level_texture_pyramid", False):
        expected_branches = 7 + int(getattr(model, "jpm_local_branches", 0))
        if (
            model.branch_count != expected_branches
            or len(model.branch_projections) != expected_branches
        ):
            raise AssertionError("Cross-level texture pyramid changed V2.42 branches")
        if (
            model.cross_level_blocks != (7, 15, 23)
            or model.cross_level_dim != 128
            or model.cross_level_scale != 0.1
            or len(model._cross_level_hooks) != 3
        ):
            raise AssertionError("Cross-level texture pyramid constants changed")
        if getattr(model, "part_routed_cross_level_texture", False):
            if (
                len(model.cross_level_part_fusions) != 3
                or len(model.cross_level_part_expansions) != 3
            ):
                raise AssertionError("Part-routed texture path inventory changed")
            nonzero_expansions = [
                torch.count_nonzero(expansion.weight).item()
                for expansion in model.cross_level_part_expansions
            ]
            if (
                args.exact_initializer
                or args.image_texture_stage_only
                or args.body_head_stage_only
                or args.head_tail_stage_only
            ):
                if not all(nonzero_expansions):
                    raise AssertionError(
                        "Exact V2.51 texture initializer was not restored"
                    )
            elif any(nonzero_expansions):
                raise AssertionError("Part-routed texture paths are not exact zero")
        elif torch.count_nonzero(model.cross_level_expansion.weight).item():
            raise AssertionError("Cross-level expansion did not start at exact zero")
    if getattr(model, "jpm_local_branches", 0):
        if args.init_checkpoint is not None:
            raise AssertionError("JPM smoke illegally loaded a competition checkpoint")
        if (
            model.jpm_local_branches != 4
            or model.jpm_shift != 5
            or model.jpm_shuffle_groups != 2
            or model.branch_count != 11
            or model.jpm_branch_start != 7
            or len(model.branch_projections) != 11
            or int(model.backbone.num_prefix_tokens) != 5
            or model.jpm_refinement_block is None
            or model.jpm_refinement_norm is None
            or model._jpm_capture_hook is None
            or model.jpm_refinement_block.attn.num_prefix_tokens != 1
        ):
            raise AssertionError("JPM inventory or fixed constants changed")
        for branch_index in range(model.jpm_branch_start, model.branch_count):
            projection = model.branch_projections[branch_index][0]
            if projection.in_features != 1280 or projection.out_features != 512:
                raise AssertionError("JPM projection geometry changed")
        base_state = model.backbone.blocks[-1].state_dict()
        local_state = model.jpm_refinement_block.state_dict()
        if base_state.keys() != local_state.keys() or any(
            not torch.equal(base_state[name], local_state[name])
            for name in base_state
        ):
            raise AssertionError("JPM block did not copy the generic final block")
        base_norm_state = model.backbone.norm.state_dict()
        local_norm_state = model.jpm_refinement_norm.state_dict()
        if base_norm_state.keys() != local_norm_state.keys() or any(
            not torch.equal(base_norm_state[name], local_norm_state[name])
            for name in base_norm_state
        ):
            raise AssertionError("JPM norm did not copy the generic final norm")
        order = model.jpm_patch_order(784, device=torch.device("cuda"))
        if (
            order.shape != (784,)
            or torch.unique(order).numel() != 784
            or int(order.min()) != 0
            or int(order.max()) != 783
            or order[:6].tolist() != [5, 397, 6, 398, 7, 399]
            or order[-2:].tolist() != [396, 4]
        ):
            raise AssertionError("JPM fixed shift/shuffle order changed")
        del order
    if getattr(model, "image_frequency_texture_side", False):
        encoder = model.image_texture_encoder
        if (
            encoder.output_dim != 128
            or encoder.stem[0].in_channels != 7
            or encoder.stem[0].out_channels != 32
            or len(model.image_texture_part_fusions) != 3
            or len(model.image_texture_part_expansions) != 3
            or any(
                torch.count_nonzero(expansion.weight).item()
                for expansion in model.image_texture_part_expansions
            )
        ):
            raise AssertionError("Image-frequency side geometry or zero start changed")
    if getattr(model, "body_to_head_distillation", False):
        if (
            model.branch_count != 7
            or len(model.head_identity_adapters) != 7
            or model.body_head_adapter_scale != 0.1
            or model.body_head_alignment_weight != 0.20
            or any(
                torch.count_nonzero(adapter.up.weight).item()
                for adapter in model.head_identity_adapters
            )
        ):
            raise AssertionError("Body-to-head adapter geometry or zero start changed")
        if not model.body_teacher_available.all():
            raise AssertionError("Smoke teacher availability was not installed")
    if getattr(model, "head_tail_expert_blocks", 0):
        if (
            model.head_tail_expert_blocks != 2
            or model.head_tail_expert_block_indices != (30, 31)
            or len(model.head_tail_expert_modules) != 2
            or len(model._head_tail_expert_hooks) != 2
        ):
            raise AssertionError("Head-tail expert inventory changed")
        for block_index, expert in zip(
            model.head_tail_expert_block_indices,
            model.head_tail_expert_modules,
            strict=True,
        ):
            base_state = model.backbone.blocks[block_index].state_dict()
            expert_state = expert.state_dict()
            if base_state.keys() != expert_state.keys() or any(
                not torch.equal(base_state[name], expert_state[name])
                for name in base_state
            ):
                raise AssertionError(
                    f"Head-tail expert block {block_index} did not copy its mother"
                )
    stage_bn_snapshots: list[tuple[torch.nn.Module, torch.Tensor, torch.Tensor]] = []
    if stage_only:
        active_prefix = (
            "head_tail_expert_modules."
            if args.head_tail_stage_only
            else (
                "head_identity_adapters."
                if args.body_head_stage_only
                else (
                    "image_texture_"
                    if args.image_texture_stage_only
                    else "cross_level_"
                )
            )
        )
        trainable_names = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not trainable_names or any(
            not name.startswith(active_prefix) for name in trainable_names
        ):
            raise AssertionError("Frozen stage exposed an incumbent parameter")
        if model.backbone.training or model.branch_necks.training:
            raise AssertionError("Frozen incumbent modules are not in evaluation mode")
        stage_bn_snapshots = [
            (module, module.running_mean.clone(), module.running_var.clone())
            for module in model.branch_necks.modules()
            if isinstance(module, torch.nn.BatchNorm1d)
        ]
    availability = torch.zeros(3, 255, dtype=torch.bool)
    availability[
        torch.as_tensor(train.part_index.to_numpy(copy=True)),
        torch.as_tensor(train.label_index.to_numpy(copy=True)),
    ] = True
    model.set_part_availability(availability)
    if hasattr(model, "set_part_counts"):
        part_counts = torch.zeros(3, 255, dtype=torch.float32)
        grouped_counts = train.groupby(["part_index", "label_index"]).size()
        for (part_index, label_index), count in grouped_counts.items():
            part_counts[int(part_index), int(label_index)] = float(count)
        model.set_part_counts(part_counts)
        if head_tail_other_enabled:
            expected_tail_mask = part_counts[0].gt(0) & part_counts[0].le(4)
            if (
                int(expected_tail_mask.sum()) != 133
                or not torch.equal(
                    model.head_tail_identity_mask.cpu(), expected_tail_mask
                )
            ):
                raise AssertionError("V6.2 fold-train tail-ID definition changed")
            tail_probe = torch.where(expected_tail_mask)[0][0].cuda()
            other_probe = torch.where(~expected_tail_mask)[0][0].cuda()
            probe_targets = model.head_identity_expert_targets(
                torch.stack([tail_probe, other_probe])
            )
            if not torch.equal(
                probe_targets,
                torch.stack(
                    [tail_probe, tail_probe.new_tensor(model.num_classes)]
                ),
            ):
                raise AssertionError("V6.2 tail-ID/OTHER target mapping changed")
    if hasattr(model, "set_class_counts"):
        class_counts = torch.bincount(
            torch.as_tensor(train.label_index.to_numpy(copy=True)),
            minlength=255,
        )
        model.set_class_counts(class_counts)
    model_ema: ModelEMA | None = None
    model_swa: ModelSWA | None = None
    model_ema_decay = 0.0
    if ema_enabled:
        model_ema_decay = math.exp(
            math.log(0.5)
            / (args.model_ema_half_life_epochs * len(loader))
        )
        model_ema = ModelEMA(model, decay=model_ema_decay, warmup_updates=0)
        current_state = model.state_dict(keep_vars=True)
        if current_state.keys() != model_ema.shadow.keys() or any(
            not torch.equal(value.detach(), model_ema.shadow[name])
            for name, value in current_state.items()
        ):
            raise AssertionError("V4.5 EMA shadow is not an exact initial copy")
        if any(value.requires_grad for value in model_ema.shadow.values()):
            raise AssertionError("V4.5 EMA shadow unexpectedly requires gradients")
    if swa_enabled:
        model_swa = ModelSWA(model, start_epoch=args.model_swa_start_epoch)
        current_state = model.state_dict(keep_vars=True)
        if current_state.keys() != model_swa.shadow.keys() or any(
            not torch.equal(value.detach(), model_swa.shadow[name])
            for name, value in current_state.items()
        ):
            raise AssertionError("V4.6 SWA shadow is not an exact initial copy")
        if any(value.requires_grad for value in model_swa.shadow.values()):
            raise AssertionError("V4.6 SWA shadow unexpectedly requires gradients")
    optimizer = build_optimizer(
        model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=0.05,
        layer_decay=0.82,
        optimizer_name="adamw",
    )
    if head_expert_enabled:
        parameter_lrs = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        expert_ids = {
            id(parameter)
            for parameter in model.head_identity_expert_parameters()
        }
        base_ids = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if not name.startswith("head_identity_expert_")
        }
        if expert_ids & base_ids or not expert_ids:
            raise AssertionError("V6.1 expert/base parameter sets overlap")
        if any(
            abs(parameter_lrs[parameter_id] - args.head_lr) > 1e-12
            for parameter_id in expert_ids
        ):
            raise AssertionError("V6.1 expert parameter missed head LR")
    if getattr(model, "continuous_geometry_conditioning", False):
        parameter_lrs = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for parameter in model.geometry_conditioning_parameters():
            if abs(parameter_lrs[id(parameter)] - args.head_lr) > 1e-12:
                raise AssertionError("Geometry conditioner missed head LR")
    if prototype_transport_enabled:
        parameter_lrs = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for parameter in (
            model.part_prototype_transport_down,
            model.part_prototype_transport_up,
        ):
            if abs(parameter_lrs[id(parameter)] - args.head_lr) > 1e-12:
                raise AssertionError("V4.7 prototype transport missed head LR")
    foreground_token_before = None
    if foreground_token_enabled:
        parameter_lrs = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if (
            abs(
                parameter_lrs[id(model.foreground_token_embedding)]
                - args.head_lr
            )
            > 1e-12
        ):
            raise AssertionError("V4.8 foreground token missed head LR")
        if torch.count_nonzero(model.foreground_token_embedding).item() != 0:
            raise AssertionError("V4.8 foreground token is not zero-initialized")
        foreground_token_before = (
            model.foreground_token_embedding.detach().clone()
        )
    if getattr(model, "foreground_auxiliary", False):
        parameter_lrs = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for parameter in model.foreground_auxiliary_head.parameters():
            if abs(parameter_lrs[id(parameter)] - args.head_lr) > 1e-12:
                raise AssertionError("Foreground auxiliary head missed head LR")
    if v3_2_enabled:
        parameter_lrs = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for bank in (model.local_part_adapters, model.final_part_adapters):
            for parameter in bank.parameters():
                if abs(parameter_lrs[id(parameter)] - args.head_lr) > 1e-12:
                    raise AssertionError("V3.2 routed adapter missed head LR")
    if stage_only and any(
        abs(float(group["lr"]) - args.head_lr) > 1e-12
        for group in optimizer.param_groups
    ):
        raise AssertionError("Texture side parameter missed the locked head LR")
    if getattr(model, "part_mlp_expert_modules", None):
        parameter_lrs = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        layer_count = model.optimizer_layer_count()
        first_block = len(model.backbone.blocks) - len(
            model.part_mlp_expert_modules
        )
        for offset, module in enumerate(model.part_mlp_expert_modules):
            probe_parameter = next(module.experts[0].parameters())
            layer_id = first_block + offset + 1
            expected_lr = args.backbone_lr * 0.82 ** (layer_count - layer_id)
            actual_lr = parameter_lrs[id(probe_parameter)]
            if abs(actual_lr - expected_lr) > 1e-12:
                raise AssertionError(
                    f"MLP expert LR {actual_lr} != expected {expected_lr}"
                )
    if getattr(model, "jpm_local_branches", 0):
        parameter_lrs = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        layer_count = model.optimizer_layer_count()
        last_layer_id = model.optimizer_layer_id(
            f"blocks.{len(model.backbone.blocks) - 1}"
        )
        expected_refinement_lr = args.backbone_lr * 0.82 ** (
            layer_count - last_layer_id
        )
        for parameter in model.jpm_refinement_parameters():
            actual_lr = parameter_lrs[id(parameter)]
            if abs(actual_lr - expected_refinement_lr) > 1e-12:
                raise AssertionError(
                    f"JPM refinement LR {actual_lr} != {expected_refinement_lr}"
                )
        for branch_index in range(model.jpm_branch_start, model.branch_count):
            for module in (
                model.branch_projections[branch_index],
                model.branch_necks[branch_index],
            ):
                for parameter in module.parameters():
                    if not parameter.requires_grad:
                        continue
                    if abs(parameter_lrs[id(parameter)] - args.head_lr) > 1e-12:
                        raise AssertionError("JPM projection/BN missed head LR")
        for parameter in (model.shared_class_weight, model.part_class_delta):
            if abs(parameter_lrs[id(parameter)] - args.head_lr) > 1e-12:
                raise AssertionError("JPM classifier slices missed head LR")
    if getattr(model, "frozen_prefix_qv_lora", False):
        parameter_lrs = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for parameter in model.prefix_qv_lora_parameters():
            if abs(parameter_lrs[id(parameter)] - args.head_lr) > 1e-12:
                raise AssertionError("Q/V LoRA parameter missed adapter/head LR")
    if getattr(model, "pattern_a2gc_branch", False):
        parameter_lrs = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for parameter in model.pattern_aggregator.parameters():
            if abs(parameter_lrs[id(parameter)] - 3e-4) > 1e-12:
                raise AssertionError("A2GC parameter did not receive head learning rate")
    if getattr(model, "hierarchical_slot_architecture", False):
        parameter_lrs = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for parameter in model.slot_aggregator.parameters():
            if abs(parameter_lrs[id(parameter)] - 3e-4) > 1e-12:
                raise AssertionError("Slot parameter did not receive head learning rate")
    amp_initial_scale = (
        1024.0
        if (
            getattr(model, "pattern_a2gc_branch", False)
            or getattr(model, "hierarchical_slot_architecture", False)
            or getattr(model, "identity_query_pooling", False)
            or getattr(model, "continuous_geometry_conditioning", False)
            or getattr(model, "dual_level_convnext", False)
        )
        else 65536.0
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=True, init_scale=amp_initial_scale
    )
    # Construct the smoke scheduler only after verifying the optimizer's
    # declared peak learning rates: LambdaLR installs its first warm-up value
    # at construction time.
    scheduler = (
        build_scheduler(optimizer, total_steps=100, warmup_steps=3)
        if sam_enabled
        else None
    )
    batch = next(iter(loader))
    if geometry_enabled:
        expected_batch_length = 10 if foreground_enabled else 6
        if len(batch) != expected_batch_length:
            raise AssertionError("V2.61 smoke batch lacks crop geometry")
        if foreground_enabled:
            (
                images,
                labels,
                parts,
                sample_indices,
                sources,
                image_geometry,
                foreground_targets,
                foreground_valid,
                foreground_augmented,
                foreground_source_jitter,
            ) = batch
        else:
            images, labels, parts, sample_indices, sources, image_geometry = batch
            foreground_targets = None
            foreground_valid = None
            foreground_augmented = None
            foreground_source_jitter = None
        if (
            image_geometry.shape
            != (args.identities_per_batch * args.images_per_identity, 2)
            or not torch.isfinite(image_geometry).all()
            or torch.allclose(
                image_geometry,
                image_geometry[:1].expand_as(image_geometry),
            )
        ):
            raise AssertionError("V2.61 standardized crop geometry is invalid")
    else:
        expected_batch_length = 9 if foreground_enabled else 5
        if len(batch) != expected_batch_length:
            raise AssertionError("Unexpected non-geometry smoke batch")
        if foreground_enabled:
            (
                images,
                labels,
                parts,
                sample_indices,
                sources,
                foreground_targets,
                foreground_valid,
                foreground_augmented,
                foreground_source_jitter,
            ) = batch
        else:
            images, labels, parts, sample_indices, sources = batch
            foreground_targets = None
            foreground_valid = None
            foreground_augmented = None
            foreground_source_jitter = None
        image_geometry = None
    if args.sampler_profile == "balanced_cross_part":
        part_counts = torch.bincount(parts, minlength=3)
        identity_counts = torch.bincount(labels, minlength=255)
        expected_first_batch_parts = (
            [22, 22, 20] if args.full_train else [22, 21, 21]
        )
        if (
            part_counts.tolist() != expected_first_batch_parts
            or int(identity_counts.gt(0).sum()) != 16
            or not torch.all(identity_counts[identity_counts.gt(0)].eq(4))
        ):
            raise AssertionError("V4.2 first-batch quota or identity geometry changed")
        by_sample = train.set_index("sample_index")
        selected_rows = by_sample.loc[sample_indices.numpy()]
        for identity, group in selected_rows.groupby("label_index"):
            available = train.loc[
                train.label_index.eq(identity), "part_index"
            ].astype(int)
            if not set(available).issubset(set(group.part_index.astype(int))):
                raise AssertionError("V4.2 discarded an available identity part")
            available_sources = train.loc[
                train.label_index.eq(identity), "source_group"
            ].astype(str)
            expected_sources = min(4, available_sources.nunique())
            if group.source_group.astype(str).nunique() != expected_sources:
                raise AssertionError("V4.2 source-diversity preference changed")
    if foreground_enabled:
        if (
            foreground_targets is None
            or foreground_valid is None
            or foreground_augmented is None
            or foreground_source_jitter is None
            or foreground_targets.shape
            != (
                args.identities_per_batch * args.images_per_identity,
                args.image_size,
                args.image_size,
            )
            or not foreground_valid.any()
            or not foreground_source_jitter.any()
            or not (~foreground_source_jitter).any()
            or torch.any(foreground_valid & foreground_source_jitter)
            or torch.any(foreground_augmented & ~foreground_valid)
            or torch.any(
                foreground_targets[~foreground_valid].flatten(1).any(dim=1)
            )
        ):
            raise AssertionError("V2.63/V2.64 foreground batch isolation changed")
        if args.foreground_mode == "sam_bg" and (
            not foreground_augmented.any()
            or not (foreground_valid & ~foreground_augmented).any()
        ):
            raise AssertionError("V2.63 smoke lacks clean/perturbed released views")
        if args.foreground_mode == "sam_view" and (
            not foreground_augmented.any()
            or not torch.equal(foreground_augmented, foreground_valid)
        ):
            raise AssertionError("V4 smoke did not transform every valid released view")
        if args.foreground_mode == "sam_part_view":
            valid_head = foreground_valid & parts.eq(0)
            valid_body = foreground_valid & parts.ne(0)
            if (
                not valid_head.any()
                or not valid_body.any()
                or not foreground_augmented[valid_head].any()
                or not (~foreground_augmented[valid_head]).any()
                or not foreground_augmented[valid_body].all()
            ):
                raise AssertionError("V4.1 training part observation route changed")
        if args.foreground_mode == "sam_aux" and foreground_augmented.any():
            raise AssertionError("V2.64 unexpectedly changed training pixels")
    images = images.cuda()
    labels = labels.cuda()
    parts = parts.cuda()
    sources = sources.cuda()
    if image_geometry is not None:
        image_geometry = image_geometry.cuda()
    if foreground_token_enabled:
        foreground_targets = foreground_targets.cuda()
        foreground_valid = foreground_valid.cuda()
    elif getattr(model, "foreground_auxiliary", False):
        foreground_targets = foreground_targets.cuda()
        foreground_valid = foreground_valid.cuda()
    if getattr(model, "part_routed_qv_lora", False) and set(
        parts.cpu().tolist()
    ) != {0, 1, 2}:
        raise AssertionError("Part-routed Q/V LoRA smoke batch lacks a part route")
    if getattr(model, "part_mlp_expert_modules", None) and set(
        parts.cpu().tolist()
    ) != {0, 1, 2}:
        raise AssertionError("MLP expert smoke batch does not contain all parts")
    if getattr(model, "pattern_a2gc_branch", False) and set(
        parts.cpu().tolist()
    ) != {0, 1, 2}:
        raise AssertionError("A2GC smoke batch does not contain all parts")
    if getattr(model, "hierarchical_slot_architecture", False) and set(
        parts.cpu().tolist()
    ) != {0, 1, 2}:
        raise AssertionError("Slot smoke batch does not contain all parts")
    if getattr(model, "identity_query_pooling", False) and set(
        parts.cpu().tolist()
    ) != {0, 1, 2}:
        raise AssertionError("Identity-query smoke batch does not contain all parts")
    if getattr(model, "part_routed_cross_level_texture", False) and set(
        parts.cpu().tolist()
    ) != {0, 1, 2}:
        raise AssertionError("Part-routed texture smoke batch does not contain all parts")
    if getattr(model, "head_tail_expert_blocks", 0) and not torch.any(parts.eq(0)):
        raise AssertionError("Head-tail expert smoke batch contains no head sample")
    if head_expert_enabled and (
        int(parts.eq(0).sum()) not in ({21, 22} if args.full_train else {22})
    ):
        raise AssertionError("V6.1 smoke batch head route count changed")
    model_rng_state = capture_model_rng_state(torch.device("cuda"))
    smoke_label_smoothing = (
        0.05 if (sam_enabled or head_expert_enabled or external_reid_enabled) else 0.0
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        model_output = (
            model(
                images,
                parts,
                labels,
                image_geometry=image_geometry,
                foreground_mask=foreground_targets,
            )
            if foreground_token_enabled
            else (
                model(images, parts, labels)
                if image_geometry is None
                else model(
                    images,
                    parts,
                    labels,
                    image_geometry=image_geometry,
                )
            )
        )
        if geometry_enabled:
            residual = model._geometry_last_residual
            if (
                residual is None
                or residual.shape != (len(images), 789, 1280)
                or not torch.isfinite(residual).all()
                or (
                    external_reid_enabled
                    and torch.count_nonzero(residual).item() == 0
                )
                or (
                    not external_reid_enabled
                    and torch.count_nonzero(residual).item() != 0
                )
            ):
                raise AssertionError(
                    "Geometry residual differs from the declared generic/external initializer"
                )
        if foreground_token_enabled:
            foreground_residual = model._foreground_token_last_residual
            if (
                foreground_residual is None
                or foreground_residual.shape != (len(images), 789, 1280)
                or torch.count_nonzero(foreground_residual).item() != 0
            ):
                raise AssertionError(
                    "V4.8 foreground token changed the zero-start function"
                )
        shared_logits, part_logits, shared, part = model_output[:4]
        if head_expert_enabled:
            if len(model_output) != 5:
                raise AssertionError("V6 training output did not expose expert logits")
            expert_logits = model_output[4]
            expert_score = model._head_identity_expert_score
            branch_cosine = model._head_identity_expert_branch_cosine
            head_count = int(parts.eq(0).sum())
            if (
                expert_logits.shape != (head_count, 7, head_expert_width)
                or branch_cosine is None
                or branch_cosine.shape != (head_count, 7, head_expert_width)
                or expert_score is None
                or expert_score.shape != (len(images), head_expert_width)
                or not torch.isfinite(expert_logits).all()
                or not torch.isfinite(expert_score).all()
                or torch.count_nonzero(expert_score[parts.ne(0)]).item() != 0
            ):
                raise AssertionError("V6 expert route geometry is invalid")
            if head_tail_other_enabled:
                allowed = torch.cat(
                    [
                        model.head_tail_identity_mask,
                        model.head_tail_identity_mask.new_ones(1),
                    ]
                )
                if not torch.all(expert_logits[..., ~allowed] == -1e4):
                    raise AssertionError("V6.2 non-tail expert logits are not masked")
            encoded_shared = torch.cat([shared, expert_score], dim=1)
            encoded_part = torch.cat([part, expert_score], dim=1)
            model.set_head_identity_expert_routing(False)
            base_score = model.inference_scores(
                encoded_shared, encoded_part, parts
            )
            model.set_head_identity_expert_routing(True)
            routed_score = model.inference_scores(
                encoded_shared, encoded_part, parts
            )
            head_mask = parts.eq(0)
            if head_tail_other_enabled:
                expected_head = base_score[head_mask].clone()
                tail_mask = model.head_tail_identity_mask
                tail_evidence = (
                    expert_score[head_mask, :255]
                    - expert_score[head_mask, 255:]
                )
                expected_head[:, tail_mask] = (
                    base_score[head_mask][:, tail_mask]
                    + 0.10 * tail_evidence[:, tail_mask]
                )
            else:
                available_head = model.part_class_available[0][None, :]
                fallback_expert = torch.where(
                    available_head,
                    expert_score[head_mask],
                    base_score[head_mask],
                )
                expected_head = (
                    0.75 * base_score[head_mask] + 0.25 * fallback_expert
                )
            if not torch.equal(
                routed_score[parts.ne(0)], base_score[parts.ne(0)]
            ) or not torch.allclose(
                routed_score[head_mask], expected_head, atol=1e-7, rtol=1e-6
            ):
                raise AssertionError("V6 fixed part route equation changed")
            unchanged_head_mask = (
                ~model.head_tail_identity_mask
                if head_tail_other_enabled
                else ~model.part_class_available[0]
            )
            if not torch.equal(
                routed_score[head_mask][:, unchanged_head_mask],
                base_score[head_mask][:, unchanged_head_mask],
            ):
                raise AssertionError("V6 unchanged head coordinates moved")
            merged_shared, merged_part = model.combine_tta_embeddings(
                encoded_shared,
                encoded_part,
                encoded_shared,
                encoded_part,
            )
            if not torch.equal(
                merged_shared[:, -head_expert_width:], expert_score
            ) or not torch.equal(
                merged_part[:, -head_expert_width:], expert_score
            ):
                raise AssertionError("V6 TTA did not average direct expert scores")
        if petface_enabled and (
            shared_logits.shape != (len(images), 255)
            or part_logits.shape != (len(images), 255)
            or shared.shape != (len(images), 512)
            or part.shape != (len(images), 512)
            or not torch.isfinite(shared).all()
            or not torch.isfinite(part).all()
        ):
            raise AssertionError("V3.0 output geometry or finiteness changed")
        if v3_convnext_enabled:
            expected_shapes = (
                (len(images), 7, 255),
                (len(images), 7, 255),
                (len(images), 7 * 512),
                (len(images), 7 * 512),
            )
            if (
                tuple(shared_logits.shape),
                tuple(part_logits.shape),
                tuple(shared.shape),
                tuple(part.shape),
            ) != expected_shapes or not all(
                torch.isfinite(value).all()
                for value in (shared_logits, part_logits, shared, part)
            ):
                raise AssertionError(
                    "V3.1/V3.2 output geometry or finiteness changed"
                )
            if model._dual_level_last_shapes != (
                (len(images), 768, 28, 28),
                (len(images), 1536, 14, 14),
            ):
                raise AssertionError("V3.1/V3.2 dual-level feature maps changed")
            if v3_2_enabled:
                expected_counts = tuple(
                    int(parts.eq(part_index).sum()) for part_index in range(3)
                )
                if (
                    not all(expected_counts)
                    or model.local_part_adapters.last_route_counts
                    != expected_counts
                    or model.final_part_adapters.last_route_counts
                    != expected_counts
                ):
                    raise AssertionError("V3.2 did not execute all three hard routes")
        if getattr(model, "frozen_prefix_qv_lora", False):
            residual_maxima = [
                module.last_residual_max_abs
                for module in model.prefix_qv_lora_modules
            ]
            if any(value is None for value in residual_maxima) or any(
                float(value) != 0.0 for value in residual_maxima
            ):
                raise AssertionError("Q/V LoRA changed the generic QKV output at zero start")
            if (
                shared_logits.shape != (len(images), 7, 255)
                or part_logits.shape != (len(images), 7, 255)
                or shared.shape != (len(images), 7 * 512)
                or part.shape != (len(images), 7 * 512)
            ):
                raise AssertionError("Q/V LoRA changed seven-branch output geometry")
        if getattr(model, "cross_level_texture_pyramid", False):
            expected_branches = model.branch_count
            if (
                shared_logits.shape != (len(images), expected_branches, 255)
                or part_logits.shape != (len(images), expected_branches, 255)
                or shared.shape != (len(images), expected_branches * 512)
                or part.shape != (len(images), expected_branches * 512)
            ):
                raise AssertionError("Cross-level/JPM output geometry is invalid")
            residual = model._cross_level_last_residual
            if residual is None or residual.shape != (len(images), 1280, 28, 28):
                raise AssertionError("Cross-level texture residual geometry is invalid")
            if (
                not (
                    args.exact_initializer
                    or args.image_texture_stage_only
                    or args.body_head_stage_only
                    or args.head_tail_stage_only
                )
                and not torch.equal(residual, torch.zeros_like(residual))
            ):
                raise AssertionError("Zero-start texture pyramid changed the base function")
            if (
                args.exact_initializer
                or args.image_texture_stage_only
                or args.body_head_stage_only
                or args.head_tail_stage_only
            ) and (
                not torch.isfinite(residual).all()
                or not torch.any(residual != 0)
            ):
                raise AssertionError("Loaded V2.51 texture residual is invalid")
            inference_score = model.inference_scores(shared, part, parts)
            neck = shared.reshape(
                -1, model.branch_count, model.embedding_dim
            )
            base_shared_cosine, base_part_cosine = model._cosines(neck, parts)
            base_shared_score, base_part_score = model._fuse_cosines(
                base_shared_cosine, base_part_cosine, parts
            )
            available = model.part_class_available[parts]
            expected_inference = torch.where(
                available,
                0.45 * base_shared_score + 0.55 * base_part_score,
                base_shared_score,
            )
            if not torch.allclose(
                inference_score, expected_inference, atol=2e-5, rtol=2e-4
            ):
                raise AssertionError("Cross-level texture changed V2.42 inference equation")
        if getattr(model, "jpm_local_branches", 0):
            if (
                model._jpm_last_group_patch_counts != (196, 196, 196, 196)
                or model._jpm_last_group_rope_counts != (196, 196, 196, 196)
                or model._jpm_last_patch_order is None
                or model._jpm_last_patch_order.shape != (784,)
                or model._jpm_last_input_tokens is not None
                or model._jpm_last_rope is not None
            ):
                raise AssertionError("JPM patch/RoPE groups were not exact 4 x 196")
        if getattr(model, "image_frequency_texture_side", False):
            image_residual = model._image_texture_last_residual
            if (
                image_residual is None
                or image_residual.shape != (len(images), 1280, 28, 28)
                or not torch.equal(
                    image_residual, torch.zeros_like(image_residual)
                )
            ):
                raise AssertionError("Image-frequency side path changed its mother at zero start")
        if getattr(model, "body_to_head_distillation", False):
            head_residual = model._body_head_last_residual
            if (
                head_residual is None
                or head_residual.shape != (len(images), 7, 512)
                or not torch.equal(
                    head_residual, torch.zeros_like(head_residual)
                )
                or not torch.equal(
                    head_residual[parts.ne(0)],
                    torch.zeros_like(head_residual[parts.ne(0)]),
                )
            ):
                raise AssertionError(
                    "Body-to-head adapter changed the frozen mother at zero start"
                )
        if getattr(model, "identity_query_pooling", False):
            expected_logits_shape = (len(images), 8, 255)
            if (
                shared_logits.shape != expected_logits_shape
                or part_logits.shape != expected_logits_shape
            ):
                raise AssertionError(
                    "Identity-query branch did not add exactly one training logit"
                )
            shared_query, part_query = model.identity_query_scores()
            if (
                shared_query.shape != (len(images), 255)
                or part_query.shape != (len(images), 255)
                or not torch.isfinite(shared_query).all()
                or not torch.isfinite(part_query).all()
                or shared_query.abs().max() > 1.0001
                or part_query.abs().max() > 1.0001
            ):
                raise AssertionError("Identity-query class scores are invalid")
            shared_mass, part_mass = model.identity_query_attention_mass()
            if not torch.allclose(
                shared_mass, torch.ones_like(shared_mass), atol=2e-4, rtol=2e-4
            ) or not torch.allclose(
                part_mass, torch.ones_like(part_mass), atol=2e-4, rtol=2e-4
            ):
                raise AssertionError("Identity-query patch probability mass is not one")
            encoded_shared = torch.cat([shared, shared_query], dim=1)
            encoded_part = torch.cat([part, part_query], dim=1)
            inference_score = model.inference_scores(
                encoded_shared, encoded_part, parts
            )
            neck = shared.reshape(
                -1, model.branch_count, model.embedding_dim
            )
            base_shared_cosine, base_part_cosine = model._cosines(neck, parts)
            base_shared_score, base_part_score = model._fuse_cosines(
                base_shared_cosine, base_part_cosine, parts
            )
            available = model.part_class_available[parts]
            base_parametric = torch.where(
                available,
                0.45 * base_shared_score + 0.55 * base_part_score,
                base_shared_score,
            )
            query_parametric = torch.where(
                available,
                0.45 * shared_query + 0.55 * part_query,
                shared_query,
            )
            expected_inference = (
                7.0 * base_parametric + query_parametric
            ) / 8.0
            if not torch.allclose(
                inference_score, expected_inference, atol=2e-5, rtol=2e-4
            ):
                raise AssertionError("Identity-query inference is not exact 7/8 + 1/8")
            merged_shared, merged_part = model.combine_tta_embeddings(
                encoded_shared,
                encoded_part,
                encoded_shared,
                encoded_part,
            )
            if not torch.equal(merged_shared[:, -255:], shared_query) or not torch.equal(
                merged_part[:, -255:], part_query
            ):
                raise AssertionError("Identity-query TTA did not average scores directly")
        if v12_3_convnext_enabled:
            model.update_instance_queue(
                shared.detach(), labels, parts, sources
            )
            (
                queue_features,
                queue_labels,
                queue_parts,
                queue_sources,
            ) = model.instance_queue_contents()
            if (
                int(model.instance_queue_size) != len(images)
                or int(model.instance_queue_pointer) != len(images)
                or queue_features.requires_grad
                or tuple(queue_features.shape) != (len(images), 7 * 512)
                or not torch.equal(queue_labels, labels)
                or not torch.equal(queue_parts, parts)
                or not torch.equal(queue_sources, sources)
                or not torch.allclose(
                    queue_features.float().norm(dim=1),
                    torch.ones(len(images), device=images.device),
                    atol=2e-3,
                    rtol=2e-3,
                )
            ):
                raise AssertionError("V12.3 queue enqueue/provenance changed")
        if part_logits.ndim == 3:
            branch_labels = labels[:, None].expand(
                -1, part_logits.shape[1]
            ).reshape(-1)
            part_ce = F.cross_entropy(
                part_logits.flatten(0, 1),
                branch_labels,
                label_smoothing=smoke_label_smoothing,
            )
            shared_ce = F.cross_entropy(
                shared_logits.flatten(0, 1),
                branch_labels,
                label_smoothing=smoke_label_smoothing,
            )
        else:
            part_ce = F.cross_entropy(
                part_logits,
                labels,
                label_smoothing=smoke_label_smoothing,
            )
            shared_ce = F.cross_entropy(
                shared_logits,
                labels,
                label_smoothing=smoke_label_smoothing,
            )
        head_expert_ce = shared.sum() * 0.0
        if head_expert_enabled:
            head_labels = labels[parts.eq(0)]
            expert_loss_logits, expert_targets = (
                model.head_identity_expert_loss_space(
                    model_output[4], head_labels
                )
            )
            if head_tail_other_enabled and expert_loss_logits.shape[-1] != 134:
                raise AssertionError("V6.2 active expert class count changed")
            expert_branch_labels = expert_targets[:, None].expand(
                -1, expert_loss_logits.shape[1]
            ).reshape(-1)
            head_expert_ce = F.cross_entropy(
                expert_loss_logits.flatten(0, 1),
                expert_branch_labels,
                label_smoothing=0.05,
            )
            base_probes = [
                next(
                    parameter
                    for parameter in reversed(list(model.backbone.parameters()))
                    if parameter.requires_grad
                ),
                model.shared_class_weight,
            ]
            leaked = torch.autograd.grad(
                head_expert_ce,
                base_probes,
                allow_unused=True,
                retain_graph=True,
            )
            if any(value is not None and torch.any(value != 0) for value in leaked):
                raise AssertionError("V6.1 expert CE leaked into the base path")
        prototype_ce = shared.sum() * 0.0
        if getattr(model, "prototype_memory", False):
            model.update_prototype_memory(shared, labels, parts)
            prototype_logits, prototype_valid = (
                model.prototype_logits_from_embedding(shared, parts, labels)
            )
            if not prototype_valid.all():
                raise AssertionError("prototype targets were not initialized")
            prototype_labels = labels[:, None].expand(
                -1, prototype_logits.shape[1]
            ).reshape(-1)
            prototype_ce = F.cross_entropy(
                prototype_logits.flatten(0, 1), prototype_labels
            )
        auxiliary = shared.sum() * 0.0
        if getattr(model, "hierarchical_slot_architecture", False):
            auxiliary = model.auxiliary_training_loss()
            attention = model.slot_aggregator.last_attention
            if attention is None or not torch.isfinite(attention).all():
                raise AssertionError("Slot competition mask is missing or non-finite")
            if torch.any(attention <= 0):
                raise AssertionError("Slot competition mask is not strictly positive")
            if not torch.allclose(
                attention.sum(dim=1),
                torch.ones_like(attention[:, 0]),
                atol=2e-3,
                rtol=2e-3,
            ):
                raise AssertionError("Slot competition mass is not one per patch")
            if not torch.isfinite(auxiliary) or float(auxiliary.detach()) <= 0.0:
                raise AssertionError("Slot reconstruction objective is invalid")
        if getattr(model, "body_to_head_distillation", False):
            auxiliary = model.auxiliary_training_loss()
            if not torch.isfinite(auxiliary) or float(auxiliary.detach()) <= 0.0:
                raise AssertionError("Body-to-head alignment objective is invalid")
        dense_match = shared.sum() * 0.0
        if getattr(model, "dense_correspondence_training", False):
            local_sets = model.dense_correspondence_tokens()
            if local_sets.shape != (len(images), 16, 128):
                raise AssertionError(
                    f"Unexpected dense local set shape: {local_sets.shape}"
                )
            dense_raw, valid_anchors = source_aware_dense_chamfer_contrastive(
                local_sets,
                labels,
                parts,
                sources,
                temperature=model.dense_correspondence_temperature,
            )
            if int(valid_anchors) <= 0:
                raise AssertionError("Dense correspondence batch has no legal anchors")
            if not torch.isfinite(dense_raw) or float(dense_raw.detach()) <= 0.0:
                raise AssertionError("Dense correspondence objective is invalid")
            normalized_local = F.normalize(local_sets.float(), dim=-1)
            patch_similarity = torch.einsum(
                "bkd,cld->bckl", normalized_local, normalized_local
            )
            chamfer = 0.5 * (
                patch_similarity.amax(dim=3).mean(dim=2)
                + patch_similarity.amax(dim=2).mean(dim=2)
            )
            if not torch.allclose(chamfer, chamfer.T, atol=1e-6, rtol=1e-5):
                raise AssertionError("Dense Chamfer score is not symmetric")
            dense_match = model.dense_correspondence_weight * dense_raw
        foreground_raw = shared.sum() * 0.0
        foreground_bce = shared.sum() * 0.0
        foreground_dice = shared.sum() * 0.0
        foreground_supervised = torch.zeros(
            (), dtype=torch.long, device=shared.device
        )
        if getattr(model, "foreground_auxiliary", False):
            (
                foreground_raw,
                foreground_bce,
                foreground_dice,
                foreground_supervised,
            ) = balanced_foreground_auxiliary(
                model.foreground_training_logits(),
                foreground_targets,
                foreground_valid,
            )
            if (
                model.foreground_training_logits().shape != (len(images), 28, 28)
                or int(foreground_supervised) <= 0
                or not torch.isfinite(foreground_raw)
                or not torch.isfinite(foreground_bce)
                or not torch.isfinite(foreground_dice)
                or float(foreground_raw.detach()) <= 0.0
                or float(foreground_bce.detach()) <= 0.0
                or float(foreground_dice.detach()) <= 0.0
            ):
                raise AssertionError("V2.64 foreground objective is invalid")
        edge_partial_ce = shared.sum() * 0.0
        edge_consistency = shared.sum() * 0.0
        if edge_partial_enabled:
            probe_sides = torch.arange(4, device=images.device)
            probe_ratios = torch.full(
                (4,), 0.20, device=images.device, dtype=torch.float32
            )
            direction_probe = edge_truncated_views(
                images[:4], probe_sides, probe_ratios
            )
            if (
                direction_probe.shape != images[:4].shape
                or not torch.isfinite(direction_probe).all()
                or torch.equal(direction_probe, images[:4])
            ):
                raise AssertionError("Four-direction edge truncation probe failed")
            side_count_tensor = torch.as_tensor(
                edge_partial_side_counts,
                dtype=torch.float32,
                device=images.device,
            )
            partial_images, selected, sampled_sides, sampled_ratios = (
                build_edge_partial_batch(
                    images,
                    parts,
                    side_count_tensor,
                    args.edge_partial_samples,
                    args.edge_partial_min_ratio,
                    args.edge_partial_max_ratio,
                )
            )
            selected_part_counts = torch.bincount(
                parts[selected], minlength=3
            )
            if (
                len(selected) != 8
                or torch.any(selected_part_counts < 2)
                or not torch.all(
                    (sampled_sides >= 0) & (sampled_sides < 4)
                )
                or float(sampled_ratios.min()) < 0.12
                or float(sampled_ratios.max()) > 0.28
                or not torch.isfinite(partial_images).all()
                or torch.equal(partial_images, images[selected])
            ):
                raise AssertionError("Part-balanced edge-partial batch is invalid")
            partial_output = model(
                partial_images,
                parts[selected],
                labels[selected],
            )
            partial_shared_logits, partial_part_logits, partial_shared = (
                partial_output[:3]
            )
            if (
                partial_shared_logits.shape != (8, 7, 255)
                or partial_part_logits.shape != (8, 7, 255)
                or partial_shared.shape != (8, 7 * 512)
            ):
                raise AssertionError("Edge-partial seven-branch geometry changed")
            partial_branch_labels = labels[selected, None].expand(
                -1, 7
            ).reshape(-1)
            partial_part_ce = F.cross_entropy(
                partial_part_logits.flatten(0, 1), partial_branch_labels
            )
            partial_shared_ce = F.cross_entropy(
                partial_shared_logits.flatten(0, 1), partial_branch_labels
            )
            edge_partial_ce = partial_part_ce + 0.5 * partial_shared_ce
            edge_consistency = branch_cosine_consistency(
                partial_shared,
                shared[selected],
                model.branch_count,
            )
            if (
                not torch.isfinite(edge_partial_ce)
                or float(edge_partial_ce.detach()) <= 0.0
                or not torch.isfinite(edge_consistency)
                or float(edge_consistency.detach()) <= 0.0
            ):
                raise AssertionError("Edge-partial objectives are invalid")
        smoke_supcon = model_supcon(
            model, shared, labels, parts, sources, 0.10
        )
        queue_gradient = None
        if v12_3_convnext_enabled:
            queue_gradient = torch.autograd.grad(
                smoke_supcon, shared, retain_graph=True
            )[0]
            if (
                not torch.isfinite(queue_gradient).all()
                or not torch.any(queue_gradient != 0)
                or model._instance_queue_last_augmented_valid <= 0
            ):
                raise AssertionError("V12.3 queued topology loss gave no signal")
        loss = (
            part_ce
            + 0.5 * shared_ce
            + 0.15 * smoke_supcon
            + 0.30 * source_aware_part_batch_hard(
                part, labels, parts, sources, 0.10
            )
            + 0.50 * prototype_ce
            + auxiliary
            + dense_match
            + args.foreground_aux_weight * foreground_raw
            + args.edge_partial_ce_weight * edge_partial_ce
            + args.edge_partial_consistency_weight * edge_consistency
            + getattr(model, "head_identity_expert_loss_weight", 0.0)
            * head_expert_ce
        )
    scaler.scale(loss).backward()
    sam_perturbation_norm = 0.0
    sam_second_loss = 0.0
    if sam_enabled:
        rng_after_first = capture_model_rng_state(torch.device("cuda"))
        perturbation = sam_first_step(
            optimizer,
            scaler,
            args.sam_rho,
            verify_actual_norm=True,
        )
        if perturbation.parameter_tensors <= 0:
            raise AssertionError("SAM perturbed no optimizer-owned tensor")
        sam_perturbation_norm = perturbation.perturbation_norm
        batch_norm_after_first = [
            (
                module,
                module.running_mean.clone() if module.running_mean is not None else None,
                module.running_var.clone() if module.running_var is not None else None,
                module.num_batches_tracked.clone()
                if module.num_batches_tracked is not None
                else None,
            )
            for module in model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        ]
        restore_model_rng_state(model_rng_state, torch.device("cuda"))
        batch_norm_states = disable_batch_norm_running_stats(model)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                second_components = basic_training_objective(
                    model,
                    images,
                    labels,
                    parts,
                    sources,
                    image_geometry,
                    label_smoothing=0.05,
                    shared_ce_weight=0.50,
                    supcon_weight=0.15,
                    triplet_weight=0.30,
                    supcon_temperature=0.10,
                    triplet_scale=0.10,
                )
            sam_second_loss = float(second_components["loss"].detach())
            if not torch.isfinite(second_components["loss"]):
                raise AssertionError("SAM second-pass loss is non-finite")
            scaler.scale(second_components["loss"]).backward()
        except BaseException:
            restore_sam_parameters(perturbation)
            perturbation.originals.clear()
            raise
        finally:
            restore_batch_norm_running_stats(batch_norm_states)
        rng_after_second = capture_model_rng_state(torch.device("cuda"))
        if not torch.equal(rng_after_first[0], rng_after_second[0]) or not torch.equal(
            rng_after_first[1], rng_after_second[1]
        ):
            raise AssertionError("SAM did not replay identical stochastic model masks")
        for module, running_mean, running_var, batches in batch_norm_after_first:
            if (
                (running_mean is not None and not torch.equal(module.running_mean, running_mean))
                or (running_var is not None and not torch.equal(module.running_var, running_var))
                or (
                    batches is not None
                    and not torch.equal(module.num_batches_tracked, batches)
                )
            ):
                raise AssertionError("SAM second pass changed BatchNorm running statistics")
        restore_sam_parameters(perturbation)
        if not all(
            torch.equal(parameter, original)
            for parameter, original in perturbation.originals
        ):
            raise AssertionError("SAM failed exact pre-AdamW parameter restoration")
        scheduler_step_before = int(scheduler.last_epoch)
        scale_before_step = scaler.get_scale()
        optimizer_ran, total_gradient_norm = sam_second_step(
            model,
            optimizer,
            scaler,
            perturbation,
            1.0,
        )
        if optimizer_ran:
            scheduler.step()
        if not optimizer_ran or int(scheduler.last_epoch) != scheduler_step_before + 1:
            raise AssertionError("SAM did not execute exactly one optimizer/scheduler step")
    else:
        scaler.unscale_(optimizer)
        total_gradient_norm = clip_model_gradients(model, 1.0)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_ran = scaler.get_scale() >= scale_before_step
    if model_ema is not None:
        if not optimizer_ran:
            raise AssertionError("V4.5 online AdamW update was skipped")
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if any(id(value) in optimizer_parameter_ids for value in model_ema.shadow.values()):
            raise AssertionError("V4.5 EMA shadow entered the optimizer")
        probe_name = None
        online_state = model.state_dict(keep_vars=True)
        for name, parameter in model.named_parameters():
            if (
                parameter.requires_grad
                and torch.is_floating_point(parameter)
                and not torch.equal(parameter.detach(), model_ema.shadow[name])
            ):
                probe_name = name
                break
        if probe_name is None:
            raise AssertionError("V4.5 found no real online parameter update")
        shadow_before = model_ema.shadow[probe_name].clone()
        online_after = online_state[probe_name].detach().clone()
        expected = shadow_before.mul(model_ema_decay).add(
            online_after, alpha=1.0 - model_ema_decay
        )
        model_ema.update(model)
        if model_ema.updates != 1 or not torch.allclose(
            model_ema.shadow[probe_name], expected, atol=1e-7, rtol=1e-6
        ):
            raise AssertionError("V4.5 first EMA blend differs from its equation")
        ema_after = model_ema.shadow[probe_name].clone()
        with model_ema.apply_to(model):
            if not torch.equal(
                model.state_dict(keep_vars=True)[probe_name].detach(), ema_after
            ):
                raise AssertionError("V4.5 EMA state was not exposed for evaluation")
        if (
            not torch.equal(
                model.state_dict(keep_vars=True)[probe_name].detach(), online_after
            )
            or not torch.equal(model_ema.shadow[probe_name], ema_after)
        ):
            raise AssertionError("V4.5 EMA evaluation swap did not restore both states")
        if any(value.grad is not None for value in model_ema.shadow.values()):
            raise AssertionError("V4.5 EMA shadow received a gradient")
    if model_swa is not None:
        if not optimizer_ran:
            raise AssertionError("V4.6 online AdamW update was skipped")
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if any(id(value) in optimizer_parameter_ids for value in model_swa.shadow.values()):
            raise AssertionError("V4.6 SWA shadow entered the optimizer")
        online_state = model.state_dict(keep_vars=True)
        probe_name = next(
            (
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and torch.is_floating_point(parameter)
                and not torch.equal(parameter.detach(), model_swa.shadow[name])
            ),
            None,
        )
        if probe_name is None:
            raise AssertionError("V4.6 found no real online parameter update")
        model_swa.update(model, human_epoch=1)
        if model_swa.updates != 1 or model_swa.snapshots != 0 or any(
            not torch.equal(value.detach(), model_swa.shadow[name])
            for name, value in online_state.items()
        ):
            raise AssertionError("V4.6 pre-start SWA did not track online exactly")
        model_swa.update(model, human_epoch=args.model_swa_start_epoch)
        if model_swa.snapshots != 1:
            raise AssertionError("V4.6 fixed start is not the first SWA snapshot")
        first_snapshot = model_swa.shadow[probe_name].clone()
        with torch.inference_mode():
            online_state[probe_name].add_(0.01)
        second_online = online_state[probe_name].detach().clone()
        expected = first_snapshot.add(second_online).mul(0.5)
        model_swa.update(model, human_epoch=args.model_swa_start_epoch + 1)
        if model_swa.snapshots != 2 or not torch.allclose(
            model_swa.shadow[probe_name], expected, atol=1e-7, rtol=1e-6
        ):
            raise AssertionError("V4.6 two-snapshot SWA differs from uniform mean")
        swa_after = model_swa.shadow[probe_name].clone()
        with model_swa.apply_to(model):
            if not torch.equal(
                model.state_dict(keep_vars=True)[probe_name].detach(), swa_after
            ):
                raise AssertionError("V4.6 SWA state was not exposed for evaluation")
        if (
            not torch.equal(
                model.state_dict(keep_vars=True)[probe_name].detach(), second_online
            )
            or not torch.equal(model_swa.shadow[probe_name], swa_after)
        ):
            raise AssertionError("V4.6 SWA evaluation swap did not restore both states")
        if any(value.grad is not None for value in model_swa.shadow.values()):
            raise AssertionError("V4.6 SWA shadow received a gradient")
    if not torch.isfinite(loss):
        raise AssertionError("non-finite loss")
    if prototype_transport_enabled:
        up = model.part_prototype_transport_up
        down = model.part_prototype_transport_down
        if (
            not optimizer_ran
            or up.grad is None
            or not torch.isfinite(up.grad).all()
            or not torch.any(up.grad != 0)
            or down.grad is None
            or not torch.isfinite(down.grad).all()
            or prototype_transport_up_before is None
            or torch.equal(up.detach(), prototype_transport_up_before)
        ):
            raise AssertionError(
                "V4.7 prototype transport missed its finite real AdamW update"
            )
    if foreground_token_enabled:
        foreground_gradient = model.foreground_token_embedding.grad
        if (
            not optimizer_ran
            or foreground_gradient is None
            or not torch.isfinite(foreground_gradient).all()
            or not torch.any(foreground_gradient != 0)
            or foreground_token_before is None
            or torch.equal(
                model.foreground_token_embedding.detach(),
                foreground_token_before,
            )
        ):
            raise AssertionError(
                "V4.8 foreground token missed its finite real AdamW update"
            )
    if head_expert_enabled:
        expert_gradients = [
            parameter.grad
            for parameter in model.head_identity_expert_parameters()
            if parameter.requires_grad
        ]
        if (
            not optimizer_ran
            or not expert_gradients
            or any(gradient is None for gradient in expert_gradients)
            or not all(
                torch.isfinite(gradient).all()
                for gradient in expert_gradients
                if gradient is not None
            )
            or not any(
                torch.any(gradient != 0)
                for gradient in expert_gradients
                if gradient is not None
            )
            or head_expert_before is None
            or torch.equal(
                model.head_identity_expert_class_weight.detach(),
                head_expert_before,
            )
        ):
            raise AssertionError(
                "V6.1 head expert missed its finite real AdamW update"
            )
    frozen = [parameter for parameter in model.backbone.parameters() if not parameter.requires_grad]
    trainable = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    if any(parameter.grad is not None for parameter in frozen):
        raise AssertionError("frozen backbone parameter received a gradient")
    if stage_only:
        if trainable:
            raise AssertionError("Frozen stage left a backbone parameter trainable")
    elif not trainable or not any(parameter.grad is not None for parameter in trainable):
        raise AssertionError("trainable backbone did not receive a gradient")
    if getattr(model, "part_adapters", None) is not None:
        adapter_parameters = [
            parameter
            for adapter in model.part_adapters
            for parameter in adapter.parameters()
        ]
        if not any(parameter.grad is not None for parameter in adapter_parameters):
            raise AssertionError("part adapter did not receive a gradient")
    if petface_enabled:
        for layer_name in ("layer2", "layer3", "layer4"):
            gradients = [
                parameter.grad
                for parameter in getattr(model.backbone, layer_name).parameters()
                if parameter.requires_grad
            ]
            if (
                not gradients
                or any(gradient is None for gradient in gradients)
                or not all(torch.isfinite(gradient).all() for gradient in gradients)
                or not any(torch.any(gradient != 0) for gradient in gradients)
            ):
                raise AssertionError(f"V3.0 {layer_name} received no finite signal")
        for name, parameter in (
            ("shared classifier", model.shared_class_weight),
            ("part classifier", model.part_class_delta),
        ):
            if (
                parameter.grad is None
                or not torch.isfinite(parameter.grad).all()
                or not torch.any(parameter.grad != 0)
            ):
                raise AssertionError(f"V3.0 {name} received no finite signal")
    if v3_convnext_enabled:
        def require_signal(name: str, parameters: list[torch.nn.Parameter]) -> None:
            gradients = [
                parameter.grad
                for parameter in parameters
                if parameter.requires_grad
            ]
            if (
                not gradients
                or any(gradient is None for gradient in gradients)
                or not all(torch.isfinite(gradient).all() for gradient in gradients)
                or not any(torch.any(gradient != 0) for gradient in gradients)
            ):
                raise AssertionError(
                    f"V3.1/V3.2 {name} received no finite signal"
                )

        require_signal("stage 2", list(model.backbone.stages[2].parameters()))
        require_signal("stage 3", list(model.backbone.stages[3].parameters()))
        require_signal(
            "covariance reduction", list(model.covariance_reduction.parameters())
        )
        for branch_index, projection in enumerate(model.branch_projections):
            require_signal(
                f"branch projection {branch_index}", list(projection.parameters())
            )
        require_signal("shared classifier", [model.shared_class_weight])
        require_signal("part classifier", [model.part_class_delta])
        if (
            not optimizer_ran
            or v3_convnext_update_probe is None
            or torch.equal(
                v3_convnext_update_probe[0], v3_convnext_update_probe[1]
            )
        ):
            raise AssertionError(
                "V3.1/V3.2 did not execute one real AdamW update"
            )
        if v12_3_convnext_enabled and (
            queue_gradient is None
            or int(model.instance_queue_size) != len(images)
            or int(model.instance_queue_pointer) != len(images)
        ):
            raise AssertionError("V12.3 queue state changed during AdamW update")
        if v3_2_enabled:
            for level_name, bank in (
                ("local", model.local_part_adapters),
                ("final", model.final_part_adapters),
            ):
                for part_index, route in enumerate(bank.routes):
                    require_signal(
                        f"{level_name} adapter part {part_index}",
                        list(route.parameters()),
                    )
            if (
                v3_2_adapter_update_probe is None
                or torch.equal(
                    v3_2_adapter_update_probe[0],
                    v3_2_adapter_update_probe[1],
                )
            ):
                raise AssertionError("V3.2 adapter did not receive a real update")
    if getattr(model, "learned_branch_gates", False):
        gate_parameters = [
            model.shared_branch_gate,
            model.part_branch_gate,
        ]
        if not all(parameter.grad is not None for parameter in gate_parameters):
            raise AssertionError("learned branch gate did not receive a gradient")
    if getattr(model, "part_side_embedding_enabled", False):
        if model.part_side_embedding.grad is None:
            raise AssertionError("part side embedding did not receive a gradient")
        if not torch.isfinite(model.part_side_embedding.grad).all():
            raise AssertionError("part side embedding gradient is non-finite")
    if getattr(model, "continuous_geometry_conditioning", False):
        output_layer = model.geometry_conditioner[-1]
        output_gradient = output_layer.weight.grad
        if (
            output_gradient is None
            or not torch.isfinite(output_gradient).all()
            or not torch.any(output_gradient != 0)
        ):
            gradient_state = (
                "none"
                if output_gradient is None
                else (
                    f"finite={bool(torch.isfinite(output_gradient).all())} "
                    f"max={float(output_gradient.abs().max())} "
                    f"nonzero={int(torch.count_nonzero(output_gradient))}"
                )
            )
            raise AssertionError(
                "Geometry conditioner output received no signal: "
                f"gradient={gradient_state}, total_grad_norm="
                f"{float(total_gradient_norm)}, optimizer_ran={optimizer_ran}, "
                f"scale={scale_before_step}"
            )
        first_layer_gradient = model.geometry_conditioner[0].weight.grad
        if first_layer_gradient is not None and not torch.isfinite(
            first_layer_gradient
        ).all():
            raise AssertionError("Geometry conditioner early-layer gradient is non-finite")
        if (
            not external_reid_enabled
            and not sam_enabled
            and first_layer_gradient is not None
            and torch.any(first_layer_gradient != 0)
        ):
            raise AssertionError(
                "Zero-start geometry conditioner propagated an early-layer gradient"
            )
        if external_reid_enabled and (
            first_layer_gradient is None
            or not torch.any(first_layer_gradient != 0)
        ):
            raise AssertionError(
                "Externally trained geometry conditioner received no early-layer signal"
            )
        if torch.count_nonzero(output_layer.weight).item() == 0:
            raise AssertionError("Geometry conditioner remained zero after optimizer step")
    if getattr(model, "foreground_auxiliary", False):
        foreground_gradients = [
            parameter.grad for parameter in model.foreground_auxiliary_head.parameters()
        ]
        if (
            any(gradient is None for gradient in foreground_gradients)
            or not all(
                torch.isfinite(gradient).all()
                for gradient in foreground_gradients
                if gradient is not None
            )
            or not any(
                torch.any(gradient != 0)
                for gradient in foreground_gradients
                if gradient is not None
            )
        ):
            raise AssertionError("Foreground auxiliary head received no finite signal")
    if getattr(model, "frozen_prefix_adaptformer", False) and not stage_only:
        adapter_parameters = model.prefix_adaptformer_parameters()
        adapter_gradients = [
            parameter.grad
            for parameter in adapter_parameters
            if parameter.grad is not None
        ]
        if not adapter_gradients:
            raise AssertionError("AdaptFormer prefix received no gradients")
        if not all(torch.isfinite(gradient).all() for gradient in adapter_gradients):
            raise AssertionError("AdaptFormer prefix gradient is non-finite")
        if not any(gradient.abs().max() > 0 for gradient in adapter_gradients):
            raise AssertionError("AdaptFormer prefix gradient is identically zero")
    if getattr(model, "frozen_prefix_qv_lora", False):
        for block_index, module in enumerate(model.prefix_qv_lora_modules):
            for path_name, up in (
                ("query", module.query_up),
                ("value", module.value_up),
            ):
                gradient = up.weight.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise AssertionError(
                        f"Q/V LoRA {path_name} up gradient is invalid at block {block_index}"
                    )
                if not torch.any(gradient != 0):
                    raise AssertionError(
                        f"Q/V LoRA {path_name} up path received no signal at block {block_index}"
                    )
            for path_name, down in (
                ("query", module.query_down),
                ("value", module.value_down),
            ):
                gradient = down.weight.grad
                if gradient is not None and (
                    not torch.isfinite(gradient).all() or torch.any(gradient != 0)
                ):
                    raise AssertionError(
                        f"Q/V LoRA {path_name} down gradient was not exact zero at block {block_index}"
                    )
            if getattr(model, "part_routed_qv_lora", False):
                for part_index in range(3):
                    for path_name, up in (
                        ("query", module.query_part_up[part_index]),
                        ("value", module.value_part_up[part_index]),
                    ):
                        gradient = up.weight.grad
                        if gradient is None or not torch.isfinite(gradient).all():
                            raise AssertionError(
                                f"Part {part_index} Q/V LoRA {path_name} up gradient "
                                f"is invalid at block {block_index}"
                            )
                        if not torch.any(gradient != 0):
                            raise AssertionError(
                                f"Part {part_index} Q/V LoRA {path_name} up path "
                                f"received no signal at block {block_index}"
                            )
                    for path_name, down in (
                        ("query", module.query_part_down[part_index]),
                        ("value", module.value_part_down[part_index]),
                    ):
                        gradient = down.weight.grad
                        if gradient is not None and (
                            not torch.isfinite(gradient).all()
                            or torch.any(gradient != 0)
                        ):
                            raise AssertionError(
                                f"Part {part_index} Q/V LoRA {path_name} down gradient "
                                f"was not exact zero at block {block_index}"
                            )
            if any(
                parameter.grad is not None
                for parameter in module.base_qkv.parameters()
            ):
                raise AssertionError(
                    f"Frozen public QKV received a gradient at block {block_index}"
                )
        for block_index in range(args.freeze_blocks, len(model.backbone.blocks)):
            suffix_gradients = [
                parameter.grad
                for parameter in model.backbone.blocks[block_index].parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not suffix_gradients or not all(
                torch.isfinite(gradient).all() for gradient in suffix_gradients
            ) or not any(torch.any(gradient != 0) for gradient in suffix_gradients):
                raise AssertionError(
                    f"Q/V LoRA trainable suffix block {block_index} received no signal"
                )
        probe_module = model.prefix_qv_lora_modules[0]
        probe_module.eval()
        with torch.no_grad():
            probe_features = torch.ones(1, 2, 1280, device="cuda")
            if getattr(model, "part_routed_qv_lora", False):
                probe_deltas = [
                    probe_module.adapter_forward(
                        probe_features,
                        torch.tensor([part_index], device="cuda"),
                    )
                    for part_index in range(3)
                ]
                if not all(
                    torch.count_nonzero(
                        probe_module.query_part_up[index].weight
                    ).item()
                    and torch.count_nonzero(
                        probe_module.value_part_up[index].weight
                    ).item()
                    for index in range(3)
                ):
                    raise AssertionError("A routed Q/V LoRA up path stayed at zero")
            else:
                probe_deltas = [probe_module.adapter_forward(probe_features)]
        probe_module.train()
        for probe_delta in probe_deltas:
            if not torch.any(probe_delta[..., :1280] != 0) or not torch.any(
                probe_delta[..., 2560:] != 0
            ):
                raise AssertionError("Q/V LoRA optimizer step left a residual path at zero")
            if not torch.equal(
                probe_delta[..., 1280:2560],
                torch.zeros_like(probe_delta[..., 1280:2560]),
            ):
                raise AssertionError("Q/V LoRA illegally modified the key slice")
    if getattr(model, "jpm_local_branches", 0):
        if args.freeze_blocks != 24:
            raise AssertionError("JPM smoke must use the declared 24-block freeze")
        gradient_groups = {
            "refinement_block": list(model.jpm_refinement_block.parameters()),
            "refinement_norm": list(model.jpm_refinement_norm.parameters()),
        }
        for branch_index in range(model.jpm_branch_start, model.branch_count):
            gradient_groups[f"projection_{branch_index}"] = list(
                model.branch_projections[branch_index].parameters()
            )
        for name, parameters in gradient_groups.items():
            gradients = [
                parameter.grad
                for parameter in parameters
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise AssertionError(f"JPM {name} has invalid gradients")
            if not any(torch.any(gradient != 0) for gradient in gradients):
                raise AssertionError(f"JPM {name} received no signal")
        for block_index in range(args.freeze_blocks, len(model.backbone.blocks)):
            suffix_gradients = [
                parameter.grad
                for parameter in model.backbone.blocks[block_index].parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not suffix_gradients or not all(
                torch.isfinite(gradient).all() for gradient in suffix_gradients
            ) or not any(torch.any(gradient != 0) for gradient in suffix_gradients):
                raise AssertionError(
                    f"Trainable suffix block {block_index} received no finite signal"
                )
    if getattr(model, "prefix_convpass_modules", None):
        adapter_parameters = model.prefix_convpass_parameters()
        adapter_gradients = [
            parameter.grad
            for parameter in adapter_parameters
            if parameter.grad is not None
        ]
        if not adapter_gradients:
            raise AssertionError("ConvPass prefix received no gradients")
        if not all(torch.isfinite(gradient).all() for gradient in adapter_gradients):
            raise AssertionError("ConvPass prefix gradient is non-finite")
        up_gradients = [
            module.adapter_up.weight.grad
            for module in model.prefix_convpass_modules
            if module.adapter_up.weight.grad is not None
        ]
        if not up_gradients or not any(
            gradient.abs().max() > 0 for gradient in up_gradients
        ):
            raise AssertionError("ConvPass up projections received no signal")
        if getattr(model, "head_convpass_attention", False):
            probe_module = model.prefix_convpass_modules[0]
            probe = torch.randn(
                2,
                probe_module.prefix_tokens + 16,
                probe_module.adapter_down.in_features,
                device="cuda",
            )
            model._part_routing_context.parts = torch.tensor(
                [0, 1], device="cuda"
            )
            probe_module.eval()
            with torch.no_grad():
                routed_delta = probe_module.routed_adapter_forward(probe)
            probe_module.train()
            model._part_routing_context.parts = None
            if not torch.equal(
                routed_delta[1], torch.zeros_like(routed_delta[1])
            ):
                raise AssertionError("Body sample received head-only ConvPass output")
            if not torch.any(routed_delta[0] != 0):
                raise AssertionError("Head sample received no routed ConvPass output")
    if getattr(model, "part_mlp_expert_modules", None):
        for block_index, module in enumerate(model.part_mlp_expert_modules):
            for part_index, expert in enumerate(module.experts):
                gradients = [
                    parameter.grad
                    for parameter in expert.parameters()
                    if parameter.grad is not None
                ]
                if not gradients or not all(
                    torch.isfinite(gradient).all() for gradient in gradients
                ):
                    raise AssertionError(
                        f"MLP expert {block_index}:{part_index} has invalid gradients"
                    )
                if not any(gradient.abs().max() > 0 for gradient in gradients):
                    raise AssertionError(
                        f"MLP expert {block_index}:{part_index} received no signal"
                    )
    if getattr(model, "pattern_a2gc_branch", False):
        aggregator = model.pattern_aggregator
        gradient_groups = {
            "local": list(aggregator.local_projection.parameters()),
            "assignment_hidden": list(aggregator.assignment_hidden.parameters()),
            "assignment_output": list(aggregator.assignment_output.parameters()),
            "global": list(aggregator.global_projection.parameters()),
            "coordinate": list(aggregator.coordinate_projection.parameters()),
            "cluster_geometry": [aggregator.cluster_geometry],
            "geometry_weight": [aggregator.geometry_weight],
            "dustbin": [aggregator.dustbin_score],
            "output_projection": list(model.branch_projections[-1].parameters()),
        }
        for name, parameters in gradient_groups.items():
            gradients = [
                parameter.grad
                for parameter in parameters
                if parameter.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise AssertionError(f"A2GC {name} path has invalid gradients")
            if not any(gradient.abs().max() > 0 for gradient in gradients):
                raise AssertionError(f"A2GC {name} path received no signal")
    if getattr(model, "hierarchical_slot_architecture", False):
        aggregator = model.slot_aggregator
        gradient_groups = {
            "intermediate_projection": list(
                aggregator.intermediate_projection.parameters()
            ),
            "final_projection": list(aggregator.final_projection.parameters()),
            "coordinate": list(aggregator.coordinate_projection.parameters()),
            "query": list(aggregator.query_projection.parameters()),
            "key": list(aggregator.key_projection.parameters()),
            "value": list(aggregator.value_projection.parameters()),
            "gru": list(aggregator.slot_update.parameters()),
            "slot_mlp": list(aggregator.slot_mlp.parameters()),
            "slot_initialization": [aggregator.slot_initialization],
            "aggregation_query": [aggregator.aggregation_query],
            "aggregation_attention": list(
                aggregator.aggregation_attention.parameters()
            ),
            "aggregation_mlp": list(aggregator.aggregation_mlp.parameters()),
            "reconstruction": list(aggregator.reconstruction_decoder.parameters()),
        }
        for branch_index in range(1, 6):
            gradient_groups[f"branch_projection_{branch_index}"] = list(
                model.branch_projections[branch_index].parameters()
            )
        for name, parameters in gradient_groups.items():
            gradients = [
                parameter.grad
                for parameter in parameters
                if parameter.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise AssertionError(f"Slot {name} path has invalid gradients")
            if not any(gradient.abs().max() > 0 for gradient in gradients):
                raise AssertionError(f"Slot {name} path received no signal")
        for part_index in range(3):
            gradient = aggregator.slot_initialization.grad[part_index]
            if not torch.isfinite(gradient).all() or not torch.any(gradient != 0):
                raise AssertionError(
                    f"Slot initialization for part {part_index} received no signal"
                )
    if getattr(model, "dense_correspondence_training", False):
        gradient_groups = {
            "intermediate_projection": list(
                model.dense_intermediate_projection.parameters()
            ),
            "final_projection": list(model.dense_final_projection.parameters()),
        }
        for name, parameters in gradient_groups.items():
            gradients = [
                parameter.grad
                for parameter in parameters
                if parameter.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise AssertionError(
                    f"Dense correspondence {name} has invalid gradients"
                )
            if not any(gradient.abs().max() > 0 for gradient in gradients):
                raise AssertionError(
                    f"Dense correspondence {name} received no signal"
                )
    if getattr(model, "identity_query_pooling", False):
        gradient_groups = {
            "patch_projection": list(
                model.identity_query_patch_projection.parameters()
            ),
            "weight_projection": list(
                model.identity_query_weight_projection.parameters()
            ),
        }
        for name, parameters in gradient_groups.items():
            gradients = [
                parameter.grad
                for parameter in parameters
                if parameter.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise AssertionError(
                    f"Identity-query {name} has invalid gradients"
                )
            if not any(gradient.abs().max() > 0 for gradient in gradients):
                raise AssertionError(
                    f"Identity-query {name} received no signal"
                )
    if (
        getattr(model, "cross_level_texture_pyramid", False)
        and not (
            args.image_texture_stage_only
            or args.body_head_stage_only
            or args.head_tail_stage_only
        )
    ):
        expansions = (
            list(model.cross_level_part_expansions)
            if getattr(model, "part_routed_cross_level_texture", False)
            else [model.cross_level_expansion]
        )
        for part_index, expansion in enumerate(expansions):
            gradient = expansion.weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise AssertionError(
                    f"Cross-level expansion {part_index} gradient is invalid"
                )
            if not torch.any(gradient != 0):
                raise AssertionError(
                    f"Cross-level expansion {part_index} received no signal"
                )
    if getattr(model, "image_frequency_texture_side", False):
        for part_index, expansion in enumerate(
            model.image_texture_part_expansions
        ):
            gradient = expansion.weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise AssertionError(
                    f"Image-frequency expansion {part_index} gradient is invalid"
                )
            if not torch.any(gradient != 0):
                raise AssertionError(
                    f"Image-frequency expansion {part_index} received no signal"
                )
    if getattr(model, "body_to_head_distillation", False):
        for branch_index, adapter in enumerate(model.head_identity_adapters):
            gradient = adapter.up.weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise AssertionError(
                    f"Body-to-head adapter {branch_index} gradient is invalid"
                )
            if not torch.any(gradient != 0):
                raise AssertionError(
                    f"Body-to-head adapter {branch_index} received no signal"
                )
    if getattr(model, "head_tail_expert_blocks", 0):
        for block_index, expert in zip(
            model.head_tail_expert_block_indices,
            model.head_tail_expert_modules,
            strict=True,
        ):
            gradients = [
                parameter.grad
                for parameter in expert.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise AssertionError(
                    f"Head-tail expert block {block_index} has invalid gradients"
                )
            if not any(gradient.abs().max() > 0 for gradient in gradients):
                raise AssertionError(
                    f"Head-tail expert block {block_index} received no signal"
                )
    if stage_only:
        side_parameters = (
            model.head_tail_expert_parameters()
            if args.head_tail_stage_only
            else (
                model.body_head_adapter_parameters()
                if args.body_head_stage_only
                else (
                    model.image_texture_parameters()
                    if args.image_texture_stage_only
                    else model.cross_level_texture_parameters()
                )
            )
        )
        side_parameter_ids = {id(parameter) for parameter in side_parameters}
        for name, parameter in model.named_parameters():
            if id(parameter) not in side_parameter_ids and parameter.grad is not None:
                raise AssertionError(
                    f"Frozen incumbent parameter received a gradient: {name}"
                )
        for module, running_mean, running_var in stage_bn_snapshots:
            if not torch.equal(module.running_mean, running_mean) or not torch.equal(
                module.running_var, running_var
            ):
                raise AssertionError("Frozen incumbent BN statistics drifted")
    if getattr(model, "prototype_memory", False):
        if not model.shared_prototype_seen[labels].all():
            raise AssertionError("shared prototype memory did not update")
        if not model.part_prototype_seen[parts, labels].all():
            raise AssertionError("part prototype memory did not update")
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    if getattr(model, "part_mlp_expert_modules", None) and peak_memory_gb >= 24.0:
        raise AssertionError(f"MLP expert smoke exceeded 24 GB: {peak_memory_gb:.2f}")
    if getattr(model, "pattern_a2gc_branch", False) and peak_memory_gb >= 24.0:
        raise AssertionError(f"A2GC smoke exceeded 24 GB: {peak_memory_gb:.2f}")
    if getattr(model, "hierarchical_slot_architecture", False) and peak_memory_gb >= 24.0:
        raise AssertionError(f"Slot smoke exceeded 24 GB: {peak_memory_gb:.2f}")
    if getattr(model, "dense_correspondence_training", False) and peak_memory_gb >= 24.0:
        raise AssertionError(
            f"Dense correspondence smoke exceeded 24 GB: {peak_memory_gb:.2f}"
        )
    if getattr(model, "identity_query_pooling", False) and peak_memory_gb >= 24.0:
        raise AssertionError(
            f"Identity-query smoke exceeded 24 GB: {peak_memory_gb:.2f}"
        )
    if getattr(model, "cross_level_texture_pyramid", False) and peak_memory_gb >= 24.0:
        raise AssertionError(
            f"Cross-level texture smoke exceeded 24 GB: {peak_memory_gb:.2f}"
        )
    if getattr(model, "image_frequency_texture_side", False) and peak_memory_gb >= 24.0:
        raise AssertionError(
            f"Image-frequency smoke exceeded 24 GB: {peak_memory_gb:.2f}"
        )
    if getattr(model, "body_to_head_distillation", False) and peak_memory_gb >= 24.0:
        raise AssertionError(
            f"Body-to-head smoke exceeded 24 GB: {peak_memory_gb:.2f}"
        )
    if getattr(model, "head_tail_expert_blocks", 0) and peak_memory_gb >= 24.0:
        raise AssertionError(
            f"Head-tail expert smoke exceeded 24 GB: {peak_memory_gb:.2f}"
        )
    if getattr(model, "jpm_local_branches", 0) and peak_memory_gb >= 24.0:
        raise AssertionError(f"JPM smoke exceeded 24 GB: {peak_memory_gb:.2f}")
    if getattr(model, "frozen_prefix_qv_lora", False) and peak_memory_gb >= 24.0:
        raise AssertionError(f"Q/V LoRA smoke exceeded 24 GB: {peak_memory_gb:.2f}")
    if edge_partial_enabled and peak_memory_gb >= 24.0:
        raise AssertionError(
            f"Edge-partial consistency smoke exceeded 24 GB: {peak_memory_gb:.2f}"
        )
    if geometry_enabled and peak_memory_gb >= 24.0:
        raise AssertionError(
            f"Continuous geometry smoke exceeded 24 GB: {peak_memory_gb:.2f}"
        )
    if petface_enabled and peak_memory_gb >= 24.0:
        raise AssertionError(f"V3.0 smoke exceeded 24 GB: {peak_memory_gb:.2f}")
    if v3_convnext_enabled and peak_memory_gb >= 24.0:
        raise AssertionError(
            f"V3.1/V3.2 smoke exceeded 24 GB: {peak_memory_gb:.2f}"
        )
    geometry_span = (
        float(image_geometry.float().std().detach())
        if image_geometry is not None
        else 0.0
    )
    print(
        f"SMOKE_OK loss={float(loss.detach()):.4f} batch={len(images)} "
        f"memory_gb={peak_memory_gb:.2f} "
        f"edge_ce={float(edge_partial_ce.detach()):.4f} "
        f"edge_cons={float(edge_consistency.detach()):.4f} "
        f"geometry_std={geometry_span:.4f} "
        f"fg_valid={int(foreground_valid.sum()) if foreground_valid is not None else 0} "
        f"fg_aug={int(foreground_augmented.sum()) if foreground_augmented is not None else 0} "
        f"fg_sup={int(foreground_supervised)} "
        f"fg_loss={float(foreground_raw.detach()):.4f} "
        f"sam_loss={sam_second_loss:.4f} sam_eps={sam_perturbation_norm:.6f} "
        f"ema_decay={model_ema_decay:.9f} "
        f"ema_updates={model_ema.updates if model_ema is not None else 0} "
        f"swa_updates={model_swa.updates if model_swa is not None else 0} "
        f"swa_snapshots={model_swa.snapshots if model_swa is not None else 0} "
        f"external_reid_loaded={int(external_reid_enabled)} "
        f"external_reid_tensors={external_reid_metadata.get('loaded_representation_tensors', 0)} "
        f"fresh_classifier={int(bool(fresh_external_class_state))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
