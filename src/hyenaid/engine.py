from __future__ import annotations

import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from decision_alignment import decision_aligned_classification
from losses import (
    balanced_foreground_auxiliary,
    source_aware_dense_chamfer_contrastive,
    source_aware_head_two_view_part_balanced_supcon,
    source_aware_part_batch_hard,
    source_aware_part_topology_queue_supcon,
    source_aware_part_topology_supcon,
    source_aware_supcon,
)


PARTS = ("head", "left_body", "right_body")


class ModelEMA:
    """Online model-state EMA with an allocation-light evaluation swap."""

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        warmup_updates: int = 0,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be strictly between zero and one")
        if warmup_updates < 0:
            raise ValueError("EMA warm-up updates cannot be negative")
        self.decay = float(decay)
        self.warmup_updates = int(warmup_updates)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict(keep_vars=True).items()
        }

    @torch.inference_mode()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict(keep_vars=True)
        if current.keys() != self.shadow.keys():
            raise AssertionError("EMA/model state keys changed")
        blend = self.updates >= self.warmup_updates
        for name, value in current.items():
            shadow = self.shadow[name]
            source = value.detach()
            if shadow.shape != source.shape or shadow.dtype != source.dtype:
                raise AssertionError(f"EMA tensor geometry changed: {name}")
            if blend and torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                shadow.copy_(source)
        self.updates += 1

    def _swap(self, model: nn.Module) -> None:
        current = model.state_dict(keep_vars=True)
        if current.keys() != self.shadow.keys():
            raise AssertionError("EMA/model state keys changed during swap")
        for name, value in current.items():
            shadow = self.shadow[name]
            temporary = value.detach().clone()
            value.copy_(shadow)
            shadow.copy_(temporary)

    @contextmanager
    def apply_to(self, model: nn.Module):
        """Temporarily expose the EMA state through the live module."""
        with torch.inference_mode():
            self._swap(model)
        try:
            yield model
        finally:
            with torch.inference_mode():
                self._swap(model)


class ModelSWA:
    """Fixed-start uniform epoch-SWA with an allocation-light eval swap."""

    def __init__(self, model: nn.Module, start_epoch: int) -> None:
        if start_epoch <= 0:
            raise ValueError("SWA start epoch must be positive")
        self.start_epoch = int(start_epoch)
        self.updates = 0
        self.snapshots = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict(keep_vars=True).items()
        }

    @torch.inference_mode()
    def update(self, model: nn.Module, human_epoch: int) -> None:
        if human_epoch <= self.updates:
            raise ValueError("SWA human epochs must increase strictly")
        current = model.state_dict(keep_vars=True)
        if current.keys() != self.shadow.keys():
            raise AssertionError("SWA/model state keys changed")
        should_average = human_epoch >= self.start_epoch
        next_snapshots = self.snapshots + 1 if should_average else 0
        for name, value in current.items():
            shadow = self.shadow[name]
            source = value.detach()
            if shadow.shape != source.shape or shadow.dtype != source.dtype:
                raise AssertionError(f"SWA tensor geometry changed: {name}")
            if should_average and self.snapshots > 0 and torch.is_floating_point(shadow):
                shadow.mul_(self.snapshots / next_snapshots).add_(
                    source, alpha=1.0 / next_snapshots
                )
            else:
                # Before the fixed start this tracks the online model exactly;
                # the first selected snapshot and non-floating buffers copy too.
                shadow.copy_(source)
        self.updates = int(human_epoch)
        if should_average:
            self.snapshots = next_snapshots

    def _swap(self, model: nn.Module) -> None:
        current = model.state_dict(keep_vars=True)
        if current.keys() != self.shadow.keys():
            raise AssertionError("SWA/model state keys changed during swap")
        for name, value in current.items():
            shadow = self.shadow[name]
            temporary = value.detach().clone()
            value.copy_(shadow)
            shadow.copy_(temporary)

    @contextmanager
    def apply_to(self, model: nn.Module):
        """Temporarily expose the SWA state through the live module."""
        with torch.inference_mode():
            self._swap(model)
        try:
            yield model
        finally:
            with torch.inference_mode():
                self._swap(model)


def model_supcon(
    model: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    parts: torch.Tensor,
    source_codes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if getattr(model, "training_instance_queue", False):
        (
            queue_embeddings,
            queue_labels,
            queue_parts,
            queue_sources,
        ) = model.instance_queue_contents()
        loss, current_valid, augmented_valid = (
            source_aware_part_topology_queue_supcon(
                embeddings,
                labels,
                parts,
                source_codes,
                queue_embeddings,
                queue_labels,
                queue_parts,
                queue_sources,
                temperature,
            )
        )
        model._instance_queue_last_current_valid = int(current_valid.item())
        model._instance_queue_last_augmented_valid = int(augmented_valid.item())
        return loss
    if getattr(model, "head_two_view_part_balanced_supcon", False):
        return source_aware_head_two_view_part_balanced_supcon(
            embeddings,
            labels,
            parts,
            source_codes,
            temperature,
        )
    if getattr(model, "part_topology_supcon", False):
        return source_aware_part_topology_supcon(
            embeddings,
            labels,
            parts,
            source_codes,
            temperature,
        )
    return source_aware_supcon(
        embeddings,
        labels,
        source_codes,
        temperature,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def part_balanced_subset_indices(
    parts: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Choose a random subset while reserving equal capacity for each part."""
    if parts.ndim != 1:
        raise ValueError("Part labels must be one-dimensional")
    sample_count = min(int(sample_count), int(parts.numel()))
    if sample_count <= 0:
        return torch.empty(0, dtype=torch.long, device=parts.device)
    base_per_part = sample_count // len(PARTS)
    chosen: list[torch.Tensor] = []
    for part_index in range(len(PARTS)):
        candidates = torch.where(parts.eq(part_index))[0]
        take = min(base_per_part, int(candidates.numel()))
        if take:
            order = torch.randperm(candidates.numel(), device=parts.device)
            chosen.append(candidates[order[:take]])
    selected = (
        torch.cat(chosen)
        if chosen
        else torch.empty(0, dtype=torch.long, device=parts.device)
    )
    remaining_count = sample_count - int(selected.numel())
    if remaining_count:
        available = torch.ones(parts.numel(), dtype=torch.bool, device=parts.device)
        available[selected] = False
        candidates = torch.where(available)[0]
        order = torch.randperm(candidates.numel(), device=parts.device)
        selected = torch.cat([selected, candidates[order[:remaining_count]]])
    order = torch.randperm(selected.numel(), device=parts.device)
    return selected[order]


def edge_truncated_views(
    images: torch.Tensor,
    sides: torch.Tensor,
    ratios: torch.Tensor,
) -> torch.Tensor:
    """Remove one outer strip per image and resize the visible remainder."""
    if images.ndim != 4 or sides.shape != ratios.shape:
        raise ValueError("Invalid edge-partial view geometry")
    if len(images) != len(sides):
        raise ValueError("Edge metadata does not match the image batch")
    if not torch.all((sides >= 0) & (sides < 4)):
        raise ValueError("Edge side must be left/top/right/bottom")
    if not torch.all((ratios > 0.0) & (ratios < 1.0)):
        raise ValueError("Edge truncation ratios must lie inside (0, 1)")
    height, width = images.shape[-2:]
    views: list[torch.Tensor] = []
    for image, side_tensor, ratio_tensor in zip(
        images, sides, ratios, strict=True
    ):
        side = int(side_tensor.item())
        ratio = float(ratio_tensor.item())
        if side in (0, 2):
            cut = min(width - 1, max(1, round(ratio * width)))
            crop = image[:, :, cut:] if side == 0 else image[:, :, : width - cut]
        else:
            cut = min(height - 1, max(1, round(ratio * height)))
            crop = image[:, cut:, :] if side == 1 else image[:, : height - cut, :]
        views.append(
            F.interpolate(
                crop.unsqueeze(0),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        )
    return torch.cat(views, dim=0)


def build_edge_partial_batch(
    images: torch.Tensor,
    parts: torch.Tensor,
    side_counts: torch.Tensor,
    sample_count: int,
    min_ratio: float,
    max_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct fold-train-profiled, part-balanced one-sided crops."""
    if side_counts.shape != (len(PARTS), 4) or torch.any(side_counts < 0):
        raise ValueError("Expected a non-negative 3 x 4 edge-count matrix")
    if not 0.0 < min_ratio <= max_ratio < 1.0:
        raise ValueError("Invalid edge-partial ratio interval")
    selected = part_balanced_subset_indices(parts, sample_count)
    if not len(selected):
        raise ValueError("Edge-partial training requires at least one sample")
    weights = side_counts.to(device=parts.device, dtype=torch.float32) + 1.0
    sides = torch.multinomial(weights[parts[selected]], 1).squeeze(1)
    ratios = torch.empty(len(selected), device=images.device).uniform_(
        min_ratio, max_ratio
    )
    partial = edge_truncated_views(images[selected], sides, ratios)
    return partial, selected, sides, ratios


def branch_cosine_consistency(
    student: torch.Tensor,
    teacher: torch.Tensor,
    branch_count: int,
) -> torch.Tensor:
    """Match corresponding branch descriptors without cross-branch collapse."""
    if student.shape != teacher.shape or student.shape[1] % branch_count:
        raise ValueError("Invalid branch descriptor geometry")
    student_branches = F.normalize(
        student.float().reshape(len(student), branch_count, -1), dim=-1
    )
    teacher_branches = F.normalize(
        teacher.detach().float().reshape(len(teacher), branch_count, -1), dim=-1
    )
    return 1.0 - (student_branches * teacher_branches).sum(dim=-1).mean()


def source_paired_body_logit_distillation(
    model: nn.Module,
    shared_embedding: torch.Tensor,
    part_embedding: torch.Tensor,
    labels: torch.Tensor,
    parts: torch.Tensor,
    source_codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match same-source body evidence into head without inference pairing."""
    raw_score_getter = getattr(model, "raw_scores_from_embeddings", None)
    if not callable(raw_score_getter):
        raise RuntimeError("Source-paired KD model has no raw score interface")
    score = raw_score_getter(shared_embedding, part_embedding, parts).float()
    head_indices: list[int] = []
    body_indices: list[int] = []
    for source in source_codes.unique():
        source_rows = torch.where(source_codes.eq(source))[0]
        heads = source_rows[parts[source_rows].eq(0)]
        bodies = source_rows[parts[source_rows].ne(0)]
        if len(heads) and len(bodies):
            if labels[source_rows].unique().numel() != 1:
                raise AssertionError("Same-source head/body labels diverged")
            head_index = int(heads[0].item())
            body_index = int(bodies[0].item())
            head_indices.append(head_index)
            body_indices.append(body_index)
    if not head_indices:
        zero = score.sum() * 0.0
        empty = torch.zeros((), dtype=torch.long, device=score.device)
        return zero, empty, empty
    head_index = torch.as_tensor(head_indices, device=score.device)
    body_index = torch.as_tensor(body_indices, device=score.device)
    pair_labels = labels[head_index]
    teacher_score = score[body_index].detach()
    teacher_correct = teacher_score.argmax(dim=1).eq(pair_labels)
    pair_count = torch.as_tensor(len(head_indices), device=score.device)
    valid_count = teacher_correct.sum()
    if not teacher_correct.any():
        return score.sum() * 0.0, pair_count, valid_count
    temperature = float(model.source_paired_logit_temperature)
    if abs(temperature - 0.10) > 1e-12:
        raise AssertionError("Source-paired KD temperature changed")
    student_log_probability = F.log_softmax(
        score[head_index[teacher_correct]] / temperature, dim=1
    )
    teacher_probability = F.softmax(
        teacher_score[teacher_correct] / temperature, dim=1
    )
    raw = F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="batchmean",
    ) * (temperature * temperature)
    return (
        float(model.source_paired_logit_weight) * raw,
        pair_count,
        valid_count,
    )


def build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    layer_decay: float,
    optimizer_name: str = "adamw",
) -> Optimizer:
    layer_count = int(model.optimizer_layer_count())
    groups: dict[tuple[str, int, bool], dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        external_head_representation = name.startswith(
            (
                "external_head_prefix_adapters.",
                "external_head_tail_blocks.",
                "external_head_norm.",
            )
        )
        jpm_refinement = name.startswith(
            ("jpm_refinement_block.", "jpm_refinement_norm.")
        )
        suffix_head_qv_lora = (
            getattr(model, "suffix_head_qv_lora", False)
            and name.startswith("backbone.blocks.")
            and ".attn.qkv." in name
            and any(
                marker in name
                for marker in (
                    ".query_down.",
                    ".query_up.",
                    ".value_down.",
                    ".value_up.",
                )
            )
        )
        if suffix_head_qv_lora:
            inner = name[len("backbone.") :]
            layer = int(model.optimizer_layer_id(inner))
            learning_rate = backbone_lr * layer_decay ** (layer_count - layer)
            family = "backbone"
        elif external_head_representation:
            if name.startswith("external_head_prefix_adapters."):
                layer = int(name.split(".")[1]) + 1
            elif name.startswith("external_head_tail_blocks."):
                layer = int(name.split(".")[1]) + 31
            else:
                layer = layer_count
            learning_rate = backbone_lr * layer_decay ** (layer_count - layer)
            family = "backbone"
        elif jpm_refinement:
            # The shared JPM block/norm are a copy of the public final
            # backbone layer and keep that layer's decayed LR, not head LR.
            layer = int(
                model.optimizer_layer_id(
                    f"blocks.{len(model.backbone.blocks) - 1}"
                )
            )
            learning_rate = backbone_lr * layer_decay ** (layer_count - layer)
            family = "backbone"
        elif name.startswith("backbone.") and not any(
            marker in name
            for marker in (
                ".adapter_down.",
                ".adapter_conv.",
                ".adapter_up.",
                ".query_down.",
                ".query_up.",
                ".value_down.",
                ".value_up.",
                ".query_part_down.",
                ".query_part_up.",
                ".value_part_down.",
                ".value_part_up.",
            )
        ):
            inner = name[len("backbone.") :]
            layer = int(model.optimizer_layer_id(inner))
            learning_rate = backbone_lr * layer_decay ** (layer_count - layer)
            family = "backbone"
        else:
            layer = layer_count + 1
            learning_rate = head_lr
            family = "head"
        key = (family, layer, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": learning_rate,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        groups[key]["params"].append(parameter)
    if optimizer_name == "adamw":
        return AdamW(list(groups.values()), betas=(0.9, 0.999))
    if optimizer_name == "adam":
        return Adam(list(groups.values()), betas=(0.9, 0.999))
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float = 0.04,
) -> LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def clip_model_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip V6.1 base/expert independently; preserve legacy global clipping."""
    expert_getter = getattr(model, "head_identity_expert_parameters", None)
    if not getattr(model, "head_identity_expert", False) or not callable(
        expert_getter
    ):
        return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
    expert_parameters = [
        parameter
        for parameter in expert_getter()
        if parameter.requires_grad and parameter.grad is not None
    ]
    expert_ids = {id(parameter) for parameter in expert_getter()}
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and id(parameter) not in expert_ids
    ]
    base_norm = torch.nn.utils.clip_grad_norm_(base_parameters, max_norm)
    expert_norm = torch.nn.utils.clip_grad_norm_(expert_parameters, max_norm)
    return math.hypot(float(base_norm), float(expert_norm))


@dataclass
class SAMPerturbationState:
    originals: list[tuple[nn.Parameter, torch.Tensor]]
    gradient_norm: float
    perturbation_norm: float
    parameter_tensors: int


def _optimizer_parameters(optimizer: Optimizer) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in seen:
                raise AssertionError("Optimizer contains a parameter more than once")
            seen.add(id(parameter))
            parameters.append(parameter)
    return parameters


def capture_model_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture only model-side CPU/CUDA RNG; input augmentation already ran."""
    if device.type != "cuda":
        raise ValueError("The locked SAM implementation requires CUDA")
    return torch.get_rng_state(), torch.cuda.get_rng_state(device)


def restore_model_rng_state(
    state: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> None:
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1], device)


def disable_batch_norm_running_stats(
    model: nn.Module,
) -> list[tuple[nn.Module, bool]]:
    """Use batch statistics without updating buffers during SAM's second pass."""
    states: list[tuple[nn.Module, bool]] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            states.append((module, bool(module.track_running_stats)))
            module.track_running_stats = False
    return states


def restore_batch_norm_running_stats(
    states: list[tuple[nn.Module, bool]],
) -> None:
    for module, track_running_stats in states:
        module.track_running_stats = track_running_stats


def sam_first_step(
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    rho: float,
    *,
    verify_actual_norm: bool = False,
) -> SAMPerturbationState:
    """Manually unscale first-pass gradients and move to the SAM adversary."""
    if rho <= 0.0:
        raise ValueError("SAM rho must be positive")
    inverse_scale = 1.0 / float(scaler.get_scale())
    parameters = [
        parameter
        for parameter in _optimizer_parameters(optimizer)
        if parameter.grad is not None
    ]
    if not parameters:
        raise AssertionError("SAM first pass produced no optimizer gradients")
    gradient_norms: list[torch.Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None or gradient.is_sparse:
            raise AssertionError("SAM requires dense gradients")
        gradient.mul_(inverse_scale)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("SAM first-pass gradient is non-finite")
        gradient_norms.append(torch.linalg.vector_norm(gradient.float(), ord=2))
    gradient_norm_tensor = torch.linalg.vector_norm(
        torch.stack(gradient_norms), ord=2
    )
    if not torch.isfinite(gradient_norm_tensor) or gradient_norm_tensor <= 0:
        raise FloatingPointError("SAM first-pass gradient norm is invalid")
    coefficient = float(rho) / (gradient_norm_tensor + 1e-12)
    originals: list[tuple[nn.Parameter, torch.Tensor]] = []
    actual_norms: list[torch.Tensor] = []
    with torch.no_grad():
        for parameter in parameters:
            original = parameter.detach().clone(
                memory_format=torch.preserve_format
            )
            originals.append((parameter, original))
            parameter.add_(parameter.grad, alpha=float(coefficient))
            if verify_actual_norm:
                actual_norms.append(
                    torch.linalg.vector_norm(
                        (parameter.detach() - original).float(), ord=2
                    )
                )
    optimizer.zero_grad(set_to_none=True)
    perturbation_norm = float(rho)
    if verify_actual_norm:
        perturbation_norm = float(
            torch.linalg.vector_norm(torch.stack(actual_norms), ord=2)
        )
        if not math.isclose(perturbation_norm, float(rho), rel_tol=2e-4, abs_tol=2e-5):
            raise AssertionError(
                f"SAM perturbation norm {perturbation_norm} != rho {rho}"
            )
    return SAMPerturbationState(
        originals=originals,
        gradient_norm=float(gradient_norm_tensor),
        perturbation_norm=perturbation_norm,
        parameter_tensors=len(parameters),
    )


def restore_sam_parameters(state: SAMPerturbationState) -> None:
    """Bit-exactly restore optimizer parameters before the base update."""
    with torch.no_grad():
        for parameter, original in state.originals:
            parameter.copy_(original)


def sam_second_step(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    state: SAMPerturbationState,
    grad_clip: float,
) -> tuple[bool, float]:
    """Restore w, then apply one base-optimizer update from grad L(w+eps)."""
    restore_sam_parameters(state)
    state.originals.clear()
    scaler.unscale_(optimizer)
    second_gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip
    )
    if not torch.isfinite(second_gradient_norm):
        raise FloatingPointError("SAM second-pass gradient norm is non-finite")
    scale_before_step = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer_ran = scaler.get_scale() >= scale_before_step
    return optimizer_ran, float(second_gradient_norm)


def basic_training_objective(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    parts: torch.Tensor,
    source_codes: torch.Tensor,
    image_geometry: torch.Tensor | None,
    *,
    label_smoothing: float,
    shared_ce_weight: float,
    supcon_weight: float,
    triplet_weight: float,
    supcon_temperature: float,
    triplet_scale: float,
) -> dict[str, torch.Tensor]:
    """The locked V2.42/V2.61 objective used on both SAM passes."""
    model_output = (
        model(images, parts, labels)
        if image_geometry is None
        else model(images, parts, labels, image_geometry=image_geometry)
    )
    if len(model_output) != 4:
        raise RuntimeError("SAM is locked to the four-output base objective")
    shared_logits, part_logits, shared_embedding, part_embedding = model_output
    if part_logits.ndim == 3:
        branch_labels = labels[:, None].expand(
            -1, part_logits.shape[1]
        ).reshape(-1)
        part_ce = F.cross_entropy(
            part_logits.flatten(0, 1),
            branch_labels,
            label_smoothing=label_smoothing,
        )
        shared_ce = F.cross_entropy(
            shared_logits.flatten(0, 1),
            branch_labels,
            label_smoothing=label_smoothing,
        )
    else:
        part_ce = F.cross_entropy(
            part_logits, labels, label_smoothing=label_smoothing
        )
        shared_ce = F.cross_entropy(
            shared_logits, labels, label_smoothing=label_smoothing
        )
    supcon = model_supcon(
        model,
        shared_embedding,
        labels,
        parts,
        source_codes,
        supcon_temperature,
    )
    triplet = source_aware_part_batch_hard(
        part_embedding,
        labels,
        parts,
        source_codes,
        scale=triplet_scale,
    )
    loss = (
        part_ce
        + shared_ce_weight * shared_ce
        + supcon_weight * supcon
        + triplet_weight * triplet
    )
    return {
        "loss": loss,
        "part_ce": part_ce,
        "shared_ce": shared_ce,
        "supcon": supcon,
        "triplet": triplet,
    }


def per_sample_classification_loss(
    shared_logits: torch.Tensor,
    part_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float,
    shared_ce_weight: float,
) -> torch.Tensor:
    """Return the exact Q classification contribution for every image row."""
    if part_logits.ndim == 3:
        batch_size, branch_count, _ = part_logits.shape
        branch_labels = labels[:, None].expand(-1, branch_count).reshape(-1)
        part_rows = F.cross_entropy(
            part_logits.flatten(0, 1),
            branch_labels,
            label_smoothing=label_smoothing,
            reduction="none",
        ).reshape(batch_size, branch_count).mean(dim=1)
        shared_rows = F.cross_entropy(
            shared_logits.flatten(0, 1),
            branch_labels,
            label_smoothing=label_smoothing,
            reduction="none",
        ).reshape(batch_size, branch_count).mean(dim=1)
    elif part_logits.ndim == 2 and shared_logits.ndim == 2:
        part_rows = F.cross_entropy(
            part_logits,
            labels,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        shared_rows = F.cross_entropy(
            shared_logits,
            labels,
            label_smoothing=label_smoothing,
            reduction="none",
        )
    else:
        raise RuntimeError("Unsupported logits geometry for per-row CE")
    return part_rows + shared_ce_weight * shared_rows


def suffix_pcgrad_named_parameters(
    model: nn.Module,
    first_block: int = 24,
    last_block: int = 31,
) -> list[tuple[str, nn.Parameter]]:
    """Select every trainable public-backbone tensor in the declared suffix."""
    selected: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not name.startswith("backbone.blocks."):
            continue
        fields = name.split(".")
        if len(fields) < 4 or not fields[2].isdigit():
            raise AssertionError(f"Unexpected backbone block parameter: {name}")
        block_index = int(fields[2])
        if first_block <= block_index <= last_block:
            selected.append((name, parameter))
    if not selected:
        raise AssertionError("No trainable suffix parameters selected for PCGrad")
    covered_blocks = {
        int(name.split(".")[2]) for name, _ in selected
    }
    expected_blocks = set(range(first_block, last_block + 1))
    if covered_blocks != expected_blocks:
        raise AssertionError(
            f"PCGrad suffix blocks {sorted(covered_blocks)} != "
            f"{sorted(expected_blocks)}"
        )
    return selected


@dataclass
class SuffixStartPointAnchorEntry:
    """One public suffix tensor and its immutable generic-pretraining start."""

    name: str
    parameter: nn.Parameter
    start_cpu: torch.Tensor
    squared_l2: float


@dataclass
class SuffixStartPointAnchor:
    """CPU-backed start-point state; deliberately outside model state."""

    entries: list[SuffixStartPointAnchorEntry]
    first_block: int
    last_block: int

    @property
    def tensor_count(self) -> int:
        return len(self.entries)

    @property
    def value_count(self) -> int:
        return sum(entry.parameter.numel() for entry in self.entries)

    @property
    def storage_bytes(self) -> int:
        return sum(
            entry.start_cpu.numel() * entry.start_cpu.element_size()
            for entry in self.entries
        )

    @property
    def initial_l2(self) -> float:
        return math.sqrt(sum(entry.squared_l2 for entry in self.entries))

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "first_block": self.first_block,
            "last_block": self.last_block,
            "tensor_count": self.tensor_count,
            "value_count": self.value_count,
            "storage_bytes": self.storage_bytes,
            "initial_l2": self.initial_l2,
            "excluded_1d_and_bias": True,
            "checkpoint_state": "excluded",
        }


@torch.no_grad()
def capture_suffix_start_point_anchor(
    model: nn.Module,
    first_block: int = 24,
    last_block: int = 31,
) -> SuffixStartPointAnchor:
    """Capture only ordinary decayed weights from public blocks 24--31."""
    entries: list[SuffixStartPointAnchorEntry] = []
    covered_blocks: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not name.startswith("backbone.blocks."):
            continue
        fields = name.split(".")
        if len(fields) < 4 or not fields[2].isdigit():
            raise AssertionError(f"Unexpected backbone block parameter: {name}")
        block_index = int(fields[2])
        if not first_block <= block_index <= last_block:
            continue
        # Mirror build_optimizer's no-decay rule exactly.
        if parameter.ndim == 1 or name.endswith(".bias"):
            continue
        start_cpu = parameter.detach().cpu().clone(
            memory_format=torch.preserve_format
        )
        entries.append(
            SuffixStartPointAnchorEntry(
                name=name,
                parameter=parameter,
                start_cpu=start_cpu,
                squared_l2=float(start_cpu.double().square().sum()),
            )
        )
        covered_blocks.add(block_index)
    expected_blocks = set(range(first_block, last_block + 1))
    if covered_blocks != expected_blocks:
        raise AssertionError(
            f"Start-point suffix blocks {sorted(covered_blocks)} != "
            f"{sorted(expected_blocks)}"
        )
    if not entries:
        raise AssertionError("No suffix start-point tensors were captured")
    return SuffixStartPointAnchor(entries, first_block, last_block)


@torch.no_grad()
def apply_suffix_start_point_anchor(
    anchor: SuffixStartPointAnchor,
    optimizer: Optimizer,
    *,
    verify_correction: bool = False,
) -> dict[str, float]:
    """Complete AdamW's decoupled decay as lr*wd*(w0-w)."""
    groups_by_parameter: dict[int, tuple[float, float]] = {}
    for group in optimizer.param_groups:
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        for parameter in group["params"]:
            key = id(parameter)
            if key in groups_by_parameter:
                raise AssertionError("Optimizer parameter appears in two groups")
            groups_by_parameter[key] = (learning_rate, weight_decay)

    squared_correction_norm = 0.0
    maximum_verification_error = 0.0
    for entry in anchor.entries:
        if id(entry.parameter) not in groups_by_parameter:
            raise AssertionError(f"Anchored tensor missing from optimizer: {entry.name}")
        learning_rate, weight_decay = groups_by_parameter[id(entry.parameter)]
        if learning_rate <= 0.0 or weight_decay <= 0.0:
            raise AssertionError(
                f"Anchored tensor has invalid AdamW group: {entry.name} "
                f"lr={learning_rate} wd={weight_decay}"
            )
        coefficient = learning_rate * weight_decay
        before = (
            entry.parameter.detach().clone(memory_format=torch.preserve_format)
            if verify_correction
            else None
        )
        start_device = entry.start_cpu.to(
            device=entry.parameter.device,
            dtype=entry.parameter.dtype,
            non_blocking=False,
        )
        entry.parameter.add_(start_device, alpha=coefficient)
        squared_correction_norm += coefficient * coefficient * entry.squared_l2
        if before is not None:
            expected = before.add(start_device, alpha=coefficient)
            maximum_verification_error = max(
                maximum_verification_error,
                float((entry.parameter - expected).abs().max()),
            )
        del start_device, before
    return {
        "suffix_anchor_updates": 1.0,
        "suffix_anchor_parameter_tensors": float(anchor.tensor_count),
        "suffix_anchor_values": float(anchor.value_count),
        "suffix_anchor_correction_norm": math.sqrt(squared_correction_norm),
        "suffix_anchor_verification_max_abs": maximum_verification_error,
    }


def head_protected_suffix_gradients(
    head_task: torch.Tensor,
    body_task: torch.Tensor,
    named_parameters: list[tuple[str, nn.Parameter]],
) -> tuple[list[torch.Tensor] | None, dict[str, float]]:
    """Measure suffix conflict and retain only the head correction direction."""
    parameters = [parameter for _, parameter in named_parameters]
    head_gradients = torch.autograd.grad(
        head_task,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    missing_head = [
        name
        for (name, _), gradient in zip(named_parameters, head_gradients)
        if gradient is None
    ]
    if missing_head:
        raise AssertionError(
            f"Head task misses {len(missing_head)} suffix tensors: "
            f"{missing_head[:3]}"
        )
    body_gradients = torch.autograd.grad(
        body_task,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    missing_body = [
        name
        for (name, _), gradient in zip(named_parameters, body_gradients)
        if gradient is None
    ]
    if missing_body:
        raise AssertionError(
            f"Body task misses {len(missing_body)} suffix tensors: "
            f"{missing_body[:3]}"
        )
    typed_head = [gradient for gradient in head_gradients if gradient is not None]
    typed_body = [gradient for gradient in body_gradients if gradient is not None]
    dot = torch.zeros((), dtype=torch.float32, device=head_task.device)
    head_norm_sq = torch.zeros_like(dot)
    body_norm_sq = torch.zeros_like(dot)
    for head_gradient, body_gradient in zip(typed_head, typed_body):
        head_float = head_gradient.float()
        body_float = body_gradient.float()
        dot = dot + torch.sum(head_float * body_float)
        head_norm_sq = head_norm_sq + torch.sum(head_float.square())
        body_norm_sq = body_norm_sq + torch.sum(body_float.square())
    if not all(
        torch.isfinite(value) for value in (dot, head_norm_sq, body_norm_sq)
    ):
        raise FloatingPointError("Non-finite V12.8 suffix task gradients")
    epsilon = torch.finfo(torch.float32).eps
    cosine = dot / torch.sqrt(
        head_norm_sq.clamp_min(epsilon) * body_norm_sq.clamp_min(epsilon)
    )
    conflict = bool(dot.item() < 0.0)
    correction_scale = (
        float((-dot / head_norm_sq.clamp_min(epsilon)).item())
        if conflict
        else 0.0
    )
    statistics = {
        "suffix_pcgrad_cosine": float(cosine.item()),
        "suffix_pcgrad_conflict_fraction": float(conflict),
        "suffix_pcgrad_correction_scale": correction_scale,
        "suffix_pcgrad_head_norm": float(torch.sqrt(head_norm_sq).item()),
        "suffix_pcgrad_body_norm": float(torch.sqrt(body_norm_sq).item()),
        "suffix_pcgrad_parameter_tensors": float(len(named_parameters)),
        "suffix_pcgrad_projected_dot": (
            float((dot + correction_scale * head_norm_sq).item())
            if conflict
            else float(dot.item())
        ),
    }
    del body_gradients, typed_body
    return (typed_head if conflict else None), statistics


@torch.no_grad()
def apply_head_protected_suffix_correction(
    named_parameters: list[tuple[str, nn.Parameter]],
    head_gradients: list[torch.Tensor] | None,
    correction_scale: float,
) -> None:
    """Replace only the conflicting body CE component after AMP unscaling."""
    if head_gradients is None:
        if correction_scale != 0.0:
            raise AssertionError("PCGrad correction has no head direction")
        return
    if not correction_scale > 0.0:
        raise AssertionError("A conflicting PCGrad correction must be positive")
    if len(named_parameters) != len(head_gradients):
        raise AssertionError("PCGrad correction geometry changed")
    for (name, parameter), head_gradient in zip(
        named_parameters, head_gradients
    ):
        if parameter.grad is None:
            raise AssertionError(f"Ordinary gradient missing for {name}")
        parameter.grad.add_(head_gradient, alpha=correction_scale)


def apply_training_patch_mask(
    images: torch.Tensor,
    ratio: float,
    patch_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mask a fixed number of aligned patches with normalized zero in train mode."""
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError("Training patch-mask ratio must be in [0, 1)")
    if patch_size <= 0:
        raise ValueError("Training patch size must be positive")
    if ratio == 0.0:
        return images, {
            "training_patch_mask_fraction": 0.0,
            "training_patch_mask_patches_per_image": 0.0,
        }
    if images.ndim != 4:
        raise ValueError("Training patch masking expects BCHW images")
    height, width = images.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by training patch size")
    grid_height = height // patch_size
    grid_width = width // patch_size
    total_patches = grid_height * grid_width
    masked_patches = int(round(total_patches * ratio))
    if masked_patches <= 0 or masked_patches >= total_patches:
        raise ValueError("Training patch-mask ratio selects an invalid patch count")
    random_order = torch.rand(
        images.shape[0], total_patches, device=images.device
    ).argsort(dim=1)
    patch_mask = torch.zeros(
        images.shape[0], total_patches, dtype=torch.bool, device=images.device
    )
    patch_mask.scatter_(1, random_order[:, :masked_patches], True)
    pixel_mask = patch_mask.reshape(
        images.shape[0], grid_height, grid_width
    ).repeat_interleave(patch_size, dim=1).repeat_interleave(
        patch_size, dim=2
    )
    masked_images = images.masked_fill(pixel_mask[:, None], 0.0)
    return masked_images, {
        "training_patch_mask_fraction": masked_patches / total_patches,
        "training_patch_mask_patches_per_image": float(masked_patches),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    label_smoothing: float,
    shared_ce_weight: float,
    supcon_weight: float,
    triplet_weight: float,
    prototype_weight: float,
    supcon_temperature: float,
    triplet_scale: float,
    grad_clip: float,
    edge_partial_samples: int = 0,
    edge_partial_min_ratio: float = 0.0,
    edge_partial_max_ratio: float = 0.0,
    edge_partial_ce_weight: float = 0.0,
    edge_partial_consistency_weight: float = 0.0,
    edge_partial_side_counts: torch.Tensor | None = None,
    sam_rho: float = 0.0,
    foreground_aux_weight: float = 0.0,
    model_ema: ModelEMA | None = None,
    suffix_head_pcgrad: bool = False,
    suffix_startpoint_anchor: SuffixStartPointAnchor | None = None,
    verify_suffix_anchor_correction: bool = False,
    train_patch_mask_ratio: float = 0.0,
    train_patch_mask_size: int = 16,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "classification": 0.0,
        "part_ce": 0.0,
        "shared_ce": 0.0,
        "decision_ce": 0.0,
        "supcon": 0.0,
        "triplet": 0.0,
        "prototype_ce": 0.0,
        "auxiliary": 0.0,
        "dense_match": 0.0,
        "edge_partial_ce": 0.0,
        "edge_consistency": 0.0,
        "sam_perturbed_loss": 0.0,
        "sam_first_gradient_norm": 0.0,
        "sam_second_gradient_norm": 0.0,
        "sam_perturbation_norm": 0.0,
        "foreground_auxiliary": 0.0,
        "foreground_bce": 0.0,
        "foreground_dice": 0.0,
        "foreground_valid_fraction": 0.0,
        "foreground_supervised_fraction": 0.0,
        "foreground_augmented_fraction": 0.0,
        "foreground_source_jitter_fraction": 0.0,
        "head_expert_ce": 0.0,
        "source_paired_logit_kd": 0.0,
        "source_paired_pairs_per_batch": 0.0,
        "source_paired_valid_pairs_per_batch": 0.0,
        "supcon_current_valid_fraction": 0.0,
        "supcon_queue_valid_fraction": 0.0,
        "instance_queue_fill_fraction": 0.0,
        "classification_decomposition_error": 0.0,
        "decision_classification_decomposition_error": 0.0,
        "suffix_pcgrad_cosine": 0.0,
        "suffix_pcgrad_conflict_fraction": 0.0,
        "suffix_pcgrad_correction_scale": 0.0,
        "suffix_pcgrad_head_norm": 0.0,
        "suffix_pcgrad_body_norm": 0.0,
        "suffix_pcgrad_parameter_tensors": 0.0,
        "suffix_pcgrad_projected_dot": 0.0,
        "suffix_anchor_updates": 0.0,
        "suffix_anchor_parameter_tensors": 0.0,
        "suffix_anchor_values": 0.0,
        "suffix_anchor_correction_norm": 0.0,
        "suffix_anchor_verification_max_abs": 0.0,
        "training_patch_mask_fraction": 0.0,
        "training_patch_mask_patches_per_image": 0.0,
    }
    if sam_rho < 0.0:
        raise ValueError("SAM rho cannot be negative")
    if sam_rho > 0.0 and (
        prototype_weight != 0.0
        or edge_partial_samples != 0
        or getattr(model, "prototype_memory", False)
        or getattr(model, "hierarchical_slot_architecture", False)
        or getattr(model, "body_to_head_distillation", False)
        or getattr(model, "dense_correspondence_training", False)
        or getattr(model, "head_identity_expert", False)
        or getattr(model, "source_paired_logit_distillation", False)
    ):
        raise ValueError("V2.62 SAM is isolated from every auxiliary mechanism")
    if suffix_head_pcgrad and (
        sam_rho != 0.0
        or prototype_weight != 0.0
        or edge_partial_samples != 0
        or foreground_aux_weight != 0.0
        or getattr(model, "prototype_memory", False)
        or getattr(model, "hierarchical_slot_architecture", False)
        or getattr(model, "body_to_head_distillation", False)
        or getattr(model, "dense_correspondence_training", False)
        or getattr(model, "head_identity_expert", False)
        or getattr(model, "source_paired_logit_distillation", False)
    ):
        raise ValueError("V12.8 PCGrad is isolated to the unmodified Q objective")
    if suffix_startpoint_anchor is not None and (
        suffix_head_pcgrad
        or sam_rho != 0.0
        or prototype_weight != 0.0
        or edge_partial_samples != 0
        or foreground_aux_weight != 0.0
        or model_ema is not None
    ):
        raise ValueError("V12.10 anchoring is isolated to the unmodified Q objective")
    decision_aligned_enabled = bool(
        getattr(model, "decision_aligned_classification", False)
    )
    if decision_aligned_enabled and (
        abs(shared_ce_weight - 0.50) > 1e-12
        or sam_rho != 0.0
        or prototype_weight != 0.0
        or edge_partial_samples != 0
        or foreground_aux_weight != 0.0
        or model_ema is not None
        or suffix_head_pcgrad
        or suffix_startpoint_anchor is not None
        or train_patch_mask_ratio != 0.0
        or getattr(model, "prototype_memory", False)
        or getattr(model, "hierarchical_slot_architecture", False)
        or getattr(model, "body_to_head_distillation", False)
        or getattr(model, "dense_correspondence_training", False)
        or getattr(model, "head_identity_expert", False)
        or getattr(model, "source_paired_logit_distillation", False)
    ):
        raise ValueError(
            "V14.2 decision alignment is isolated to the exact Q objective"
        )
    seen = 0
    for batch in loader:
        foreground_targets = None
        foreground_valid = None
        foreground_augmented = None
        foreground_source_jitter = None
        if len(batch) == 10:
            (
                images,
                labels,
                parts,
                _,
                source_codes,
                image_geometry,
                foreground_targets,
                foreground_valid,
                foreground_augmented,
                foreground_source_jitter,
            ) = batch
        elif len(batch) == 9:
            (
                images,
                labels,
                parts,
                _,
                source_codes,
                foreground_targets,
                foreground_valid,
                foreground_augmented,
                foreground_source_jitter,
            ) = batch
            image_geometry = None
        elif len(batch) == 6:
            images, labels, parts, _, source_codes, image_geometry = batch
        elif len(batch) == 5:
            images, labels, parts, _, source_codes = batch
            image_geometry = None
        else:
            raise RuntimeError(f"Unexpected training batch length: {len(batch)}")
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        parts = parts.to(device, non_blocking=True)
        source_codes = source_codes.to(device, non_blocking=True)
        if image_geometry is not None:
            image_geometry = image_geometry.to(device, non_blocking=True)
        images, patch_mask_statistics = apply_training_patch_mask(
            images,
            train_patch_mask_ratio,
            train_patch_mask_size,
        )
        foreground_token_conditioning = getattr(
            model, "foreground_token_conditioning", False
        )
        foreground_scale_normalization = getattr(
            model, "foreground_scale_normalization", False
        )
        ordered_head_grid_expert = getattr(
            model, "ordered_head_grid_expert", False
        )
        foreground_mask_conditioning = (
            foreground_token_conditioning
            or foreground_scale_normalization
            or ordered_head_grid_expert
        )
        if foreground_mask_conditioning:
            if foreground_targets is None or foreground_valid is None:
                raise AssertionError(
                    "Foreground model conditioning requires aligned SAM masks"
                )
            foreground_targets = foreground_targets.to(
                device, non_blocking=True
            )
        if getattr(model, "foreground_auxiliary", False):
            if (
                foreground_targets is None
                or foreground_valid is None
                or abs(foreground_aux_weight - 0.20) > 1e-12
            ):
                raise AssertionError("V2.64 foreground supervision differs from SPEC")
            if not foreground_token_conditioning:
                foreground_targets = foreground_targets.to(
                    device, non_blocking=True
                )
            foreground_valid = foreground_valid.to(device, non_blocking=True)
        elif foreground_aux_weight != 0.0:
            raise ValueError("Foreground loss requires its declared model head")
        optimizer.zero_grad(set_to_none=True)
        if sam_rho > 0.0:
            rng_state = capture_model_rng_state(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                first_components = basic_training_objective(
                    model,
                    images,
                    labels,
                    parts,
                    source_codes,
                    image_geometry,
                    label_smoothing=label_smoothing,
                    shared_ce_weight=shared_ce_weight,
                    supcon_weight=supcon_weight,
                    triplet_weight=triplet_weight,
                    supcon_temperature=supcon_temperature,
                    triplet_scale=triplet_scale,
                )
            first_values = {
                key: float(value.detach())
                for key, value in first_components.items()
            }
            if not torch.isfinite(first_components["loss"]):
                raise FloatingPointError("Non-finite SAM first-pass loss")
            scaler.scale(first_components["loss"]).backward()
            perturbation = sam_first_step(optimizer, scaler, sam_rho)
            restore_model_rng_state(rng_state, device)
            batch_norm_states = disable_batch_norm_running_stats(model)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    second_components = basic_training_objective(
                        model,
                        images,
                        labels,
                        parts,
                        source_codes,
                        image_geometry,
                        label_smoothing=label_smoothing,
                        shared_ce_weight=shared_ce_weight,
                        supcon_weight=supcon_weight,
                        triplet_weight=triplet_weight,
                        supcon_temperature=supcon_temperature,
                        triplet_scale=triplet_scale,
                    )
                if not torch.isfinite(second_components["loss"]):
                    raise FloatingPointError("Non-finite SAM second-pass loss")
                second_loss = float(second_components["loss"].detach())
                scaler.scale(second_components["loss"]).backward()
            except BaseException:
                restore_sam_parameters(perturbation)
                perturbation.originals.clear()
                raise
            finally:
                restore_batch_norm_running_stats(batch_norm_states)
            optimizer_ran, second_gradient_norm = sam_second_step(
                model,
                optimizer,
                scaler,
                perturbation,
                grad_clip,
            )
            if optimizer_ran:
                scheduler.step()
                if model_ema is not None:
                    model_ema.update(model)
            batch_size = len(images)
            seen += batch_size
            for key in ("loss", "part_ce", "shared_ce", "supcon", "triplet"):
                totals[key] += first_values[key] * batch_size
            totals["sam_perturbed_loss"] += second_loss * batch_size
            totals["sam_first_gradient_norm"] += (
                perturbation.gradient_norm * batch_size
            )
            totals["sam_second_gradient_norm"] += second_gradient_norm * batch_size
            totals["sam_perturbation_norm"] += (
                perturbation.perturbation_norm * batch_size
            )
            continue
        pcgrad_named_parameters: list[tuple[str, nn.Parameter]] = []
        pcgrad_head_gradients: list[torch.Tensor] | None = None
        pcgrad_statistics = {
            "classification_decomposition_error": 0.0,
            "suffix_pcgrad_cosine": 0.0,
            "suffix_pcgrad_conflict_fraction": 0.0,
            "suffix_pcgrad_correction_scale": 0.0,
            "suffix_pcgrad_head_norm": 0.0,
            "suffix_pcgrad_body_norm": 0.0,
            "suffix_pcgrad_parameter_tensors": 0.0,
            "suffix_pcgrad_projected_dot": 0.0,
        }
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model_output = (
                model(
                    images,
                    parts,
                    labels,
                    image_geometry=image_geometry,
                    foreground_mask=foreground_targets,
                )
                if foreground_mask_conditioning
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
            shared_logits, part_logits, shared_embedding, part_embedding = (
                model_output[:4]
            )
            prototype_logits = None
            prototype_valid = None
            head_expert_logits = None
            if len(model_output) == 6:
                prototype_logits, prototype_valid = model_output[4:]
            elif (
                len(model_output) == 5
                and getattr(model, "head_identity_expert", False)
            ):
                head_expert_logits = model_output[4]
            elif len(model_output) != 4:
                raise RuntimeError(
                    f"Unexpected training output length: {len(model_output)}"
                )
            decision_ce = shared_embedding.sum() * 0.0
            decision_classification_decomposition_error = (
                shared_embedding.sum() * 0.0
            )
            if decision_aligned_enabled:
                if (
                    len(model_output) != 4
                    or shared_logits.ndim != 3
                    or part_logits.shape != shared_logits.shape
                    or shared_logits.shape[1] != 7
                    or shared_logits.shape[2]
                    != model.part_class_available.shape[1]
                ):
                    raise RuntimeError(
                        "V14.2 requires matching seven-branch logits [B,7,C]"
                    )
                (
                    classification,
                    part_ce,
                    shared_ce,
                    decision_ce,
                    _decision_logits,
                ) = decision_aligned_classification(
                    shared_logits,
                    part_logits,
                    labels,
                    parts,
                    model.part_class_available,
                    label_smoothing=label_smoothing,
                )
                expected_classification = (
                    0.50 * part_ce
                    + 0.25 * shared_ce
                    + 0.75 * decision_ce
                )
                decision_classification_decomposition_error = torch.abs(
                    classification - expected_classification
                )
                if (
                    not torch.isfinite(decision_ce)
                    or not torch.isfinite(classification)
                    or float(
                        decision_classification_decomposition_error.detach()
                    )
                    != 0.0
                ):
                    raise FloatingPointError(
                        "V14.2 decision classification is invalid"
                    )
            elif part_logits.ndim == 3:
                branch_labels = labels[:, None].expand(
                    -1, part_logits.shape[1]
                ).reshape(-1)
                part_ce = F.cross_entropy(
                    part_logits.flatten(0, 1),
                    branch_labels,
                    label_smoothing=label_smoothing,
                )
                shared_ce = F.cross_entropy(
                    shared_logits.flatten(0, 1),
                    branch_labels,
                    label_smoothing=label_smoothing,
                )
            else:
                part_ce = F.cross_entropy(
                    part_logits, labels, label_smoothing=label_smoothing
                )
                shared_ce = F.cross_entropy(
                    shared_logits, labels, label_smoothing=label_smoothing
                )
            if not decision_aligned_enabled:
                classification = part_ce + shared_ce_weight * shared_ce
            supcon = model_supcon(
                model,
                shared_embedding,
                labels,
                parts,
                source_codes,
                supcon_temperature,
            )
            triplet = source_aware_part_batch_hard(
                part_embedding,
                labels,
                parts,
                source_codes,
                scale=triplet_scale,
            )
            prototype_ce = shared_embedding.sum() * 0.0
            if (
                prototype_logits is not None
                and prototype_valid is not None
                and prototype_valid.any()
            ):
                if prototype_logits.ndim != 3:
                    raise RuntimeError(
                        "Prototype logits must be branch-wise [B, M, C]"
                    )
                valid_logits = prototype_logits[prototype_valid]
                valid_labels = labels[prototype_valid]
                memory_labels = valid_labels[:, None].expand(
                    -1, valid_logits.shape[1]
                ).reshape(-1)
                prototype_ce = F.cross_entropy(
                    valid_logits.flatten(0, 1),
                    memory_labels,
                    label_smoothing=label_smoothing,
                )
            auxiliary = shared_embedding.sum() * 0.0
            if (
                getattr(model, "hierarchical_slot_architecture", False)
                or getattr(model, "body_to_head_distillation", False)
            ):
                auxiliary = model.auxiliary_training_loss()
            dense_match = shared_embedding.sum() * 0.0
            if getattr(model, "dense_correspondence_training", False):
                dense_raw, _ = source_aware_dense_chamfer_contrastive(
                    model.dense_correspondence_tokens(),
                    labels,
                    parts,
                    source_codes,
                    temperature=model.dense_correspondence_temperature,
                )
                dense_match = model.dense_correspondence_weight * dense_raw
            foreground_raw = shared_embedding.sum() * 0.0
            foreground_bce = shared_embedding.sum() * 0.0
            foreground_dice = shared_embedding.sum() * 0.0
            foreground_supervised = torch.zeros(
                (), device=shared_embedding.device, dtype=torch.long
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
            foreground_auxiliary = foreground_aux_weight * foreground_raw
            source_paired_logit_kd = shared_embedding.sum() * 0.0
            source_paired_pairs = torch.zeros(
                (), device=shared_embedding.device, dtype=torch.long
            )
            source_paired_valid_pairs = source_paired_pairs.clone()
            if getattr(model, "source_paired_logit_distillation", False):
                (
                    source_paired_logit_kd,
                    source_paired_pairs,
                    source_paired_valid_pairs,
                ) = source_paired_body_logit_distillation(
                    model,
                    shared_embedding,
                    part_embedding,
                    labels,
                    parts,
                    source_codes,
                )
            head_expert_ce = shared_embedding.sum() * 0.0
            if head_expert_logits is not None:
                head_labels = labels[parts.eq(0)]
                if len(head_labels) != len(head_expert_logits):
                    raise AssertionError("Head expert logits/labels differ")
                if len(head_labels):
                    loss_space_getter = getattr(
                        model, "head_identity_expert_loss_space", None
                    )
                    expert_loss_logits, expert_targets = (
                        loss_space_getter(head_expert_logits, head_labels)
                        if callable(loss_space_getter)
                        else (head_expert_logits, head_labels)
                    )
                    expert_branch_labels = expert_targets[:, None].expand(
                        -1, expert_loss_logits.shape[1]
                    ).reshape(-1)
                    head_expert_ce = F.cross_entropy(
                        expert_loss_logits.flatten(0, 1),
                        expert_branch_labels,
                        label_smoothing=label_smoothing,
                    )
            loss = (
                classification
                + supcon_weight * supcon
                + triplet_weight * triplet
                + prototype_weight * prototype_ce
                + auxiliary
                + dense_match
                + foreground_auxiliary
                + source_paired_logit_kd
                + getattr(model, "head_identity_expert_loss_weight", 0.0)
                * head_expert_ce
            )
            edge_partial_ce = shared_embedding.sum() * 0.0
            edge_consistency = shared_embedding.sum() * 0.0
            if edge_partial_samples > 0:
                if edge_partial_side_counts is None:
                    raise AssertionError("Missing fold-train edge-side profile")
                partial_images, selected, _, _ = build_edge_partial_batch(
                    images,
                    parts,
                    edge_partial_side_counts,
                    edge_partial_samples,
                    edge_partial_min_ratio,
                    edge_partial_max_ratio,
                )
                partial_output = (
                    model(
                        partial_images,
                        parts[selected],
                        labels[selected],
                    )
                    if image_geometry is None
                    else model(
                        partial_images,
                        parts[selected],
                        labels[selected],
                        image_geometry=image_geometry[selected],
                    )
                )
                partial_shared_logits, partial_part_logits, partial_shared = (
                    partial_output[:3]
                )
                partial_labels = labels[selected]
                if partial_part_logits.ndim == 3:
                    partial_branch_labels = partial_labels[:, None].expand(
                        -1, partial_part_logits.shape[1]
                    ).reshape(-1)
                    partial_part_ce = F.cross_entropy(
                        partial_part_logits.flatten(0, 1),
                        partial_branch_labels,
                        label_smoothing=label_smoothing,
                    )
                    partial_shared_ce = F.cross_entropy(
                        partial_shared_logits.flatten(0, 1),
                        partial_branch_labels,
                        label_smoothing=label_smoothing,
                    )
                else:
                    partial_part_ce = F.cross_entropy(
                        partial_part_logits,
                        partial_labels,
                        label_smoothing=label_smoothing,
                    )
                    partial_shared_ce = F.cross_entropy(
                        partial_shared_logits,
                        partial_labels,
                        label_smoothing=label_smoothing,
                    )
                edge_partial_ce = partial_part_ce + (
                    shared_ce_weight * partial_shared_ce
                )
                edge_consistency = branch_cosine_consistency(
                    partial_shared,
                    shared_embedding[selected],
                    int(model.branch_count),
                )
                loss = (
                    loss
                    + edge_partial_ce_weight * edge_partial_ce
                    + edge_partial_consistency_weight * edge_consistency
                )
            if suffix_head_pcgrad:
                if len(model_output) != 4:
                    raise AssertionError("V12.8 requires Q's four-output graph")
                classification_rows = per_sample_classification_loss(
                    shared_logits,
                    part_logits,
                    labels,
                    label_smoothing=label_smoothing,
                    shared_ce_weight=shared_ce_weight,
                )
                head_mask = parts.eq(0)
                body_mask = ~head_mask
                if not head_mask.any() or not body_mask.any():
                    raise AssertionError(
                        "V12.8 balanced batch must contain head and body rows"
                    )
                batch_denominator = float(len(labels))
                head_classification_task = (
                    classification_rows[head_mask].sum() / batch_denominator
                )
                body_classification_task = (
                    classification_rows[body_mask].sum() / batch_denominator
                )
                original_classification = part_ce + shared_ce_weight * shared_ce
                decomposition_error = torch.abs(
                    head_classification_task
                    + body_classification_task
                    - original_classification
                )
                if float(decomposition_error.detach()) > 2e-5:
                    raise AssertionError(
                        "V12.8 classification split no longer equals Q: "
                        f"{float(decomposition_error.detach())}"
                    )
                pcgrad_named_parameters = suffix_pcgrad_named_parameters(model)
                (
                    pcgrad_head_gradients,
                    gradient_statistics,
                ) = head_protected_suffix_gradients(
                    head_classification_task,
                    body_classification_task,
                    pcgrad_named_parameters,
                )
                pcgrad_statistics.update(gradient_statistics)
                pcgrad_statistics["classification_decomposition_error"] = float(
                    decomposition_error.detach()
                )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {float(loss)}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if suffix_head_pcgrad:
            apply_head_protected_suffix_correction(
                pcgrad_named_parameters,
                pcgrad_head_gradients,
                pcgrad_statistics["suffix_pcgrad_correction_scale"],
            )
        clip_model_gradients(model, grad_clip)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # GradScaler can skip the first optimizer step while calibrating its
        # scale. Do not advance the LR schedule when no parameter update ran.
        optimizer_ran = scaler.get_scale() >= scale_before_step
        anchor_statistics: dict[str, float] = {}
        if optimizer_ran:
            if suffix_startpoint_anchor is not None:
                anchor_statistics = apply_suffix_start_point_anchor(
                    suffix_startpoint_anchor,
                    optimizer,
                    verify_correction=verify_suffix_anchor_correction,
                )
            scheduler.step()
            if model_ema is not None:
                model_ema.update(model)
            update_memory = getattr(model, "update_prototype_memory", None)
            if update_memory is not None:
                update_memory(shared_embedding, labels, parts)
            if getattr(model, "training_instance_queue", False):
                model.update_instance_queue(
                    shared_embedding, labels, parts, source_codes
                )
        batch_size = len(images)
        seen += batch_size
        for key, value in (
            ("loss", loss),
            ("classification", classification),
            ("part_ce", part_ce),
            ("shared_ce", shared_ce),
            ("decision_ce", decision_ce),
            ("supcon", supcon),
            ("triplet", triplet),
            ("prototype_ce", prototype_ce),
            ("auxiliary", auxiliary),
            ("dense_match", dense_match),
            ("foreground_auxiliary", foreground_auxiliary),
            ("foreground_bce", foreground_bce),
            ("foreground_dice", foreground_dice),
            ("head_expert_ce", head_expert_ce),
            ("source_paired_logit_kd", source_paired_logit_kd),
            ("edge_partial_ce", edge_partial_ce),
            ("edge_consistency", edge_consistency),
            (
                "decision_classification_decomposition_error",
                decision_classification_decomposition_error,
            ),
        ):
            totals[key] += float(value.detach()) * batch_size
        totals["source_paired_pairs_per_batch"] += int(
            source_paired_pairs.item()
        ) * batch_size
        totals["source_paired_valid_pairs_per_batch"] += int(
            source_paired_valid_pairs.item()
        ) * batch_size
        for key, value in pcgrad_statistics.items():
            totals[key] += float(value) * batch_size
        for key, value in anchor_statistics.items():
            totals[key] += float(value) * batch_size
        for key, value in patch_mask_statistics.items():
            totals[key] += float(value) * batch_size
        if foreground_valid is not None:
            totals["foreground_valid_fraction"] += int(
                foreground_valid.sum().item()
            )
        if foreground_augmented is not None:
            totals["foreground_augmented_fraction"] += int(
                foreground_augmented.sum().item()
            )
        if foreground_source_jitter is not None:
            totals["foreground_source_jitter_fraction"] += int(
                foreground_source_jitter.sum().item()
            )
        totals["foreground_supervised_fraction"] += int(
            foreground_supervised.item()
        )
        if getattr(model, "training_instance_queue", False):
            totals["supcon_current_valid_fraction"] += int(
                model._instance_queue_last_current_valid
            )
            totals["supcon_queue_valid_fraction"] += int(
                model._instance_queue_last_augmented_valid
            )
            totals["instance_queue_fill_fraction"] += (
                int(model.instance_queue_size.item())
                / model.instance_queue_capacity
                * batch_size
            )
    return {key: value / max(1, seen) for key, value in totals.items()}


@torch.inference_mode()
def extract(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_flip: bool,
    return_local: bool,
    local_grid: int = 6,
) -> dict[str, torch.Tensor]:
    model.eval()
    output: dict[str, list[torch.Tensor]] = {
        "sample_index": [],
        "label_index": [],
        "part_index": [],
        "shared_embedding": [],
        "part_embedding": [],
        "classifier_score": [],
    }
    if return_local:
        output["local"] = []
    for batch in loader:
        foreground_mask = None
        if len(batch) == 7:
            (
                images,
                labels,
                parts,
                sample_indices,
                _,
                image_geometry,
                foreground_mask,
            ) = batch
        elif len(batch) == 6:
            images, labels, parts, sample_indices, _, image_geometry = batch
        elif len(batch) == 5:
            images, labels, parts, sample_indices, _ = batch
            image_geometry = None
        else:
            raise RuntimeError(f"Unexpected evaluation batch length: {len(batch)}")
        images = images.to(device, non_blocking=True)
        part_device = parts.to(device, non_blocking=True)
        geometry_device = (
            image_geometry.to(device, non_blocking=True)
            if image_geometry is not None
            else None
        )
        foreground_device = (
            foreground_mask.to(device, non_blocking=True)
            if foreground_mask is not None
            else None
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            encode_kwargs = {
                "return_local": return_local,
                "local_grid": local_grid,
            }
            if geometry_device is not None:
                encode_kwargs["image_geometry"] = geometry_device
            if foreground_device is not None:
                encode_kwargs["foreground_mask"] = foreground_device
            shared, part, local = model.encode(
                images,
                part_device,
                **encode_kwargs,
            )
            if tta_flip:
                flip_kwargs = dict(encode_kwargs)
                if foreground_device is not None:
                    flip_kwargs["foreground_mask"] = torch.flip(
                        foreground_device, dims=(2,)
                    )
                flip_shared, flip_part, flip_local = model.encode(
                    torch.flip(images, dims=(3,)),
                    part_device,
                    **flip_kwargs,
                )
                combine_embeddings = getattr(
                    model, "combine_tta_embeddings", None
                )
                if callable(combine_embeddings):
                    shared, part = combine_embeddings(
                        shared, part, flip_shared, flip_part
                    )
                else:
                    shared = F.normalize(shared + flip_shared, dim=-1)
                    part = F.normalize(part + flip_part, dim=-1)
                if return_local:
                    assert local is not None and flip_local is not None
                    side = math.isqrt(local.shape[1])
                    flip_local = torch.flip(
                        flip_local.reshape(-1, side, side, flip_local.shape[-1]),
                        dims=(2,),
                    ).flatten(1, 2)
                    local = F.normalize(local + flip_local, dim=-1)
            scores = model.inference_scores(shared, part, part_device)
        output["sample_index"].append(sample_indices.cpu())
        output["label_index"].append(labels.cpu())
        output["part_index"].append(parts.cpu())
        output["shared_embedding"].append(shared.float().cpu())
        output["part_embedding"].append(part.float().cpu())
        output["classifier_score"].append(scores.float().cpu())
        if return_local:
            assert local is not None
            output["local"].append(local.half().cpu())
    return {key: torch.cat(values, dim=0) for key, values in output.items()}


def competition_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    per_part: dict[str, dict[str, float | int]] = {}
    for part in PARTS:
        subset = frame.loc[frame.part == part]
        if subset.empty:
            raise ValueError(f"Missing part {part}")
        true = subset.individual_id.astype(str)
        predicted = subset.predicted_id.astype(str)
        per_part[part] = {
            "macro_f1": float(
                f1_score(true, predicted, average="macro", zero_division=0)
            ),
            "top1_accuracy": float(accuracy_score(true, predicted)),
            "n_samples": int(len(subset)),
            "n_true_ids": int(true.nunique()),
            "n_predicted_ids": int(predicted.nunique()),
        }
    part_f1 = [float(per_part[part]["macro_f1"]) for part in PARTS]
    part_accuracy = [float(per_part[part]["top1_accuracy"]) for part in PARTS]
    return {
        "final_score": float(np.mean(part_f1)),
        "mean_part_top1_accuracy": float(np.mean(part_accuracy)),
        "sample_weighted_top1_accuracy": float(
            accuracy_score(frame.individual_id.astype(str), frame.predicted_id.astype(str))
        ),
        "per_part": per_part,
    }


def prediction_frame(
    manifest: pd.DataFrame,
    features: dict[str, torch.Tensor],
    scores: torch.Tensor,
    labels: list[str],
) -> pd.DataFrame:
    predicted = scores.argmax(dim=1).cpu().numpy()
    frame = pd.DataFrame(
        {
            "sample_index": features["sample_index"].cpu().numpy(),
            "predicted_id": [labels[index] for index in predicted],
            "confidence": scores.max(dim=1).values.cpu().numpy(),
        }
    )
    return manifest.merge(frame, on="sample_index", validate="one_to_one")


@torch.inference_mode()
def classifier_metrics(
    model: nn.Module,
    loader: DataLoader,
    manifest: pd.DataFrame,
    labels: list[str],
    device: torch.device,
    tta_flip: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    features = extract(
        model,
        loader,
        device,
        tta_flip=tta_flip,
        return_local=False,
    )
    frame = prediction_frame(manifest, features, features["classifier_score"], labels)
    return competition_metrics(frame), frame


def _class_density(
    similarities: torch.Tensor,
    gallery_labels: torch.Tensor,
    num_classes: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.full(
        (similarities.shape[0], num_classes),
        -1.0,
        dtype=similarities.dtype,
        device=similarities.device,
    )
    available = torch.zeros(num_classes, dtype=torch.bool, device=similarities.device)
    for label in gallery_labels.unique(sorted=True):
        index = int(label)
        values = similarities[:, gallery_labels == label]
        scores[:, index] = temperature * (
            torch.logsumexp(values / temperature, dim=1) - math.log(values.shape[1])
        )
        available[index] = True
    return scores, available


@torch.inference_mode()
def gallery_fused_scores(
    train_features: dict[str, torch.Tensor],
    query_features: dict[str, torch.Tensor],
    num_classes: int,
    device: torch.device,
    temperature: float = 0.05,
    candidate_images: int = 40,
    local_weight: float = 0.40,
) -> torch.Tensor:
    train_shared = F.normalize(train_features["shared_embedding"], dim=-1).to(device)
    train_part_embedding = F.normalize(train_features["part_embedding"], dim=-1).to(device)
    train_local = F.normalize(train_features["local"].float(), dim=-1).to(device)
    train_labels = train_features["label_index"].long().to(device)
    train_parts = train_features["part_index"].long().to(device)
    query_shared = F.normalize(query_features["shared_embedding"], dim=-1).to(device)
    query_part_embedding = F.normalize(query_features["part_embedding"], dim=-1).to(device)
    query_local = F.normalize(query_features["local"].float(), dim=-1).to(device)
    query_parts = query_features["part_index"].long().to(device)

    shared_density, _ = _class_density(
        query_shared @ train_shared.T,
        train_labels,
        num_classes,
        temperature,
    )
    density = shared_density - 0.02
    for part in range(3):
        query_mask = query_parts == part
        gallery_mask = train_parts == part
        if not query_mask.any() or not gallery_mask.any():
            continue
        part_density, available = _class_density(
            query_part_embedding[query_mask] @ train_part_embedding[gallery_mask].T,
            train_labels[gallery_mask],
            num_classes,
            temperature,
        )
        query_indices = torch.where(query_mask)[0]
        class_indices = torch.where(available)[0]
        density[query_indices[:, None], class_indices[None, :]] = part_density[:, available]

    local_class = density.clone()
    keep_local = max(1, query_local.shape[1] // 2)
    for part in range(3):
        query_indices = torch.where(query_parts == part)[0]
        gallery_indices = torch.where(train_parts == part)[0]
        if not len(query_indices) or not len(gallery_indices):
            continue
        gallery_part_embedding = train_part_embedding[gallery_indices]
        for query_index in query_indices:
            image_similarity = query_part_embedding[query_index] @ gallery_part_embedding.T
            count = min(candidate_images, len(gallery_indices))
            selected_positions = image_similarity.topk(count).indices
            selected_gallery = gallery_indices[selected_positions]
            patch_similarity = torch.einsum(
                "ld,kmd->klm",
                query_local[query_index],
                train_local[selected_gallery],
            )
            query_to_gallery = patch_similarity.max(dim=2).values
            gallery_to_query = patch_similarity.max(dim=1).values
            local_similarity = 0.5 * (
                query_to_gallery.topk(keep_local, dim=1).values.mean(dim=1)
                + gallery_to_query.topk(keep_local, dim=1).values.mean(dim=1)
            )
            combined = (
                (1.0 - local_weight) * image_similarity[selected_positions]
                + local_weight * local_similarity
            )
            selected_labels = train_labels[selected_gallery]
            for label in selected_labels.unique(sorted=True):
                label_index = int(label)
                local_class[query_index, label_index] = combined[
                    selected_labels == label
                ].max()

    retrieval = 0.65 * density + 0.35 * local_class
    classifier = query_features["classifier_score"].float().to(device)
    return (0.45 * classifier + 0.55 * retrieval).cpu()


def save_checkpoint(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
