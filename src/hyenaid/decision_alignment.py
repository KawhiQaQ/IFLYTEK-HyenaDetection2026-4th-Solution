"""Training-only objective helpers aligned to the deployed branch score."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def decision_aligned_logits(
    shared_logits: torch.Tensor,
    part_logits: torch.Tensor,
    parts: torch.Tensor,
    part_class_available: torch.Tensor,
    *,
    shared_weight: float = 0.45,
    part_weight: float = 0.55,
) -> torch.Tensor:
    """Apply the inference-time seven-branch and shared/part score equation.

    Inputs are margin-adjusted training logits.  Classes unavailable within a
    released crop part fall back to the shared classifier exactly as inference
    does.  No statistic is fitted or inferred from the current batch.
    """
    if shared_logits.ndim != 3 or part_logits.shape != shared_logits.shape:
        raise ValueError("Expected matching shared/part logits [B,M,C]")
    batch, branches, classes = shared_logits.shape
    if branches < 1:
        raise ValueError("At least one decision branch is required")
    if parts.shape != (batch,):
        raise ValueError("Part indices must have shape [B]")
    if part_class_available.ndim != 2 or part_class_available.shape[1] != classes:
        raise ValueError("Availability must have shape [num_parts,C]")
    if not 0.0 <= shared_weight <= 1.0 or not 0.0 <= part_weight <= 1.0:
        raise ValueError("Decision weights must be non-negative convex weights")
    if abs(shared_weight + part_weight - 1.0) > 1e-12:
        raise ValueError("Decision weights must sum to one")
    if parts.dtype != torch.long:
        raise ValueError("Part indices must be torch.long")
    if parts.numel() and (int(parts.min()) < 0 or int(parts.max()) >= len(part_class_available)):
        raise ValueError("Part index is outside the availability table")

    shared_mean = shared_logits.mean(dim=1)
    part_mean = part_logits.mean(dim=1)
    available = part_class_available.index_select(0, parts).bool()
    combined = shared_weight * shared_mean + part_weight * part_mean
    return torch.where(available, combined, shared_mean)


def decision_aligned_classification(
    shared_logits: torch.Tensor,
    part_logits: torch.Tensor,
    labels: torch.Tensor,
    parts: torch.Tensor,
    part_class_available: torch.Tensor,
    *,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return V14.2's scale-preserving branch/deployed classification terms."""
    if labels.shape != parts.shape:
        raise ValueError("Labels and parts must have matching shape")
    branch_labels = labels[:, None].expand(-1, shared_logits.shape[1]).reshape(-1)
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
    fused_logits = decision_aligned_logits(
        shared_logits,
        part_logits,
        parts,
        part_class_available,
    )
    decision_ce = F.cross_entropy(
        fused_logits,
        labels,
        label_smoothing=label_smoothing,
    )
    classification = 0.50 * part_ce + 0.25 * shared_ce + 0.75 * decision_ce
    return classification, part_ce, shared_ce, decision_ce, fused_logits
