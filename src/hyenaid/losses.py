from __future__ import annotations

import torch
import torch.nn.functional as F


def balanced_foreground_auxiliary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-image class-balanced BCE plus soft Dice on reliable masks only."""
    if logits.ndim != 3:
        raise ValueError(f"Expected foreground logits [B,H,W], got {logits.shape}")
    if targets.ndim != 3 or len(targets) != len(logits):
        raise ValueError(f"Expected foreground targets [B,H,W], got {targets.shape}")
    if valid.shape != (len(logits),):
        raise ValueError(f"Expected foreground valid flags [B], got {valid.shape}")
    target = F.interpolate(
        targets[:, None].float(),
        size=logits.shape[-2:],
        mode="nearest",
    ).squeeze(1)
    target = target.clamp(0.0, 1.0)
    flat_target = target.flatten(1)
    flat_logits = logits.float().flatten(1)
    positives = flat_target.sum(dim=1)
    negatives = (1.0 - flat_target).sum(dim=1)
    supervised = valid.bool() & positives.gt(0) & negatives.gt(0)
    supervised_count = supervised.sum()
    if not supervised.any():
        zero = logits.sum() * 0.0
        return zero, zero, zero, supervised_count
    positive_bce = (
        F.softplus(-flat_logits) * flat_target
    ).sum(dim=1) / positives.clamp_min(1.0)
    negative_bce = (
        F.softplus(flat_logits) * (1.0 - flat_target)
    ).sum(dim=1) / negatives.clamp_min(1.0)
    balanced_bce = 0.5 * (positive_bce + negative_bce)
    probability = flat_logits.sigmoid()
    dice = 1.0 - (
        2.0 * (probability * flat_target).sum(dim=1) + 1.0
    ) / (
        probability.sum(dim=1) + positives + 1.0
    )
    bce_mean = balanced_bce[supervised].mean()
    dice_mean = dice[supervised].mean()
    return 0.5 * (bce_mean + dice_mean), bce_mean, dice_mean, supervised_count


def source_aware_supcon(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    source_codes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings.float(), dim=-1)
    similarities = embeddings @ embeddings.T / temperature
    same_identity = labels[:, None].eq(labels[None, :])
    different_source = source_codes[:, None].ne(source_codes[None, :])
    positive_mask = same_identity & different_source
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    denominator_mask = ~self_mask
    valid_anchor = positive_mask.any(dim=1)
    if not valid_anchor.any():
        return embeddings.sum() * 0.0
    logits = similarities - similarities.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * denominator_mask
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive = (log_probability * positive_mask).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_positive[valid_anchor].mean()


def source_aware_part_topology_supcon(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    parts: torch.Tensor,
    source_codes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Keep head and body identity graphs separate while joining both flanks.

    Same-identity pairs from the same released source image, and identity pairs
    crossing the head/body boundary, are neutral rather than false negatives.
    """
    if parts.shape != labels.shape:
        raise ValueError(f"Part labels {parts.shape} != identities {labels.shape}")
    if temperature <= 0.0:
        raise ValueError("SupCon temperature must be positive")
    embeddings = F.normalize(embeddings.float(), dim=-1)
    similarities = embeddings @ embeddings.T / temperature
    same_identity = labels[:, None].eq(labels[None, :])
    different_identity = ~same_identity
    different_source = source_codes[:, None].ne(source_codes[None, :])
    same_part = parts[:, None].eq(parts[None, :])
    both_body = parts[:, None].gt(0) & parts[None, :].gt(0)
    positive_mask = same_identity & different_source & (same_part | both_body)
    candidate_mask = different_identity | positive_mask
    valid_anchor = positive_mask.any(dim=1)
    if not valid_anchor.any():
        return embeddings.sum() * 0.0
    logits = similarities - similarities.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * candidate_mask
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    mean_positive = (
        (log_probability * positive_mask).sum(dim=1)
        / positive_mask.sum(dim=1).clamp_min(1)
    )
    return -mean_positive[valid_anchor].mean()


def source_aware_part_topology_queue_supcon(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    parts: torch.Tensor,
    source_codes: torch.Tensor,
    queue_embeddings: torch.Tensor,
    queue_labels: torch.Tensor,
    queue_parts: torch.Tensor,
    queue_source_codes: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """V4.3 topology SupCon with detached preceding train instances as keys.

    Only current rows are anchors. Current rows remain differentiable as both
    anchors and in-batch keys; queued rows are detached keys and can never
    receive gradients. Same-identity/same-source and head/body pairs remain
    neutral exactly as in the incumbent topology loss.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected current embeddings [B,D], got {embeddings.shape}")
    if labels.shape != parts.shape or labels.shape != source_codes.shape:
        raise ValueError("Current topology metadata shapes differ")
    if queue_embeddings.ndim != 2 or queue_embeddings.shape[1] != embeddings.shape[1]:
        raise ValueError("Queue embedding geometry differs from current embeddings")
    queue_rows = len(queue_embeddings)
    if any(
        value.shape != (queue_rows,)
        for value in (queue_labels, queue_parts, queue_source_codes)
    ):
        raise ValueError("Queue topology metadata shapes differ")
    if temperature <= 0.0:
        raise ValueError("SupCon temperature must be positive")
    if queue_embeddings.requires_grad:
        raise ValueError("Queued keys must be detached")

    current = F.normalize(embeddings.float(), dim=-1)
    queued = F.normalize(queue_embeddings.detach().float(), dim=-1)
    candidates = torch.cat((current, queued), dim=0)
    candidate_labels = torch.cat((labels, queue_labels), dim=0)
    candidate_parts = torch.cat((parts, queue_parts), dim=0)
    candidate_sources = torch.cat((source_codes, queue_source_codes), dim=0)

    same_identity = labels[:, None].eq(candidate_labels[None, :])
    different_identity = ~same_identity
    different_source = source_codes[:, None].ne(candidate_sources[None, :])
    same_part = parts[:, None].eq(candidate_parts[None, :])
    both_body = parts[:, None].gt(0) & candidate_parts[None, :].gt(0)
    positive_mask = same_identity & different_source & (same_part | both_body)
    candidate_mask = different_identity | positive_mask

    # Candidate rows begin with the current batch. Explicitly exclude each
    # anchor's own key even though its source rule already makes it neutral.
    row = torch.arange(len(labels), device=labels.device)
    positive_mask[row, row] = False
    candidate_mask[row, row] = False
    current_valid_count = positive_mask[:, : len(labels)].any(dim=1).sum()
    augmented_valid = positive_mask.any(dim=1)
    augmented_valid_count = augmented_valid.sum()
    if not augmented_valid.any():
        return current.sum() * 0.0, current_valid_count, augmented_valid_count

    logits = current @ candidates.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * candidate_mask
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    mean_positive = (
        (log_probability * positive_mask).sum(dim=1)
        / positive_mask.sum(dim=1).clamp_min(1)
    )
    return (
        -mean_positive[augmented_valid].mean(),
        current_valid_count,
        augmented_valid_count,
    )


def source_aware_head_two_view_part_balanced_supcon(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    parts: torch.Tensor,
    source_codes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Topology SupCon with head two-view positives and equal part reduction."""
    if parts.shape != labels.shape:
        raise ValueError(f"Part labels {parts.shape} != identities {labels.shape}")
    if temperature <= 0.0:
        raise ValueError("SupCon temperature must be positive")
    embeddings = F.normalize(embeddings.float(), dim=-1)
    similarities = embeddings @ embeddings.T / temperature
    same_identity = labels[:, None].eq(labels[None, :])
    different_identity = ~same_identity
    different_source = source_codes[:, None].ne(source_codes[None, :])
    same_part = parts[:, None].eq(parts[None, :])
    both_body = parts[:, None].gt(0) & parts[None, :].gt(0)
    both_head = parts[:, None].eq(0) & parts[None, :].eq(0)
    non_self = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    cross_source_positive = different_source & (same_part | both_body)
    head_two_view_positive = (~different_source) & both_head & non_self
    positive_mask = same_identity & (
        cross_source_positive | head_two_view_positive
    )
    candidate_mask = different_identity | positive_mask
    valid_anchor = positive_mask.any(dim=1)
    if not valid_anchor.any():
        return embeddings.sum() * 0.0
    logits = similarities - similarities.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * candidate_mask
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    per_anchor = -(
        (log_probability * positive_mask).sum(dim=1)
        / positive_mask.sum(dim=1).clamp_min(1)
    )
    per_part = []
    for part_index in range(3):
        selected = valid_anchor & parts.eq(part_index)
        if selected.any():
            per_part.append(per_anchor[selected].mean())
    if not per_part:
        return embeddings.sum() * 0.0
    return torch.stack(per_part).mean()


def source_aware_part_batch_hard(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    parts: torch.Tensor,
    source_codes: torch.Tensor,
    scale: float = 0.10,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings.float(), dim=-1)
    distances = 1.0 - embeddings @ embeddings.T
    same_identity = labels[:, None].eq(labels[None, :])
    same_part = parts[:, None].eq(parts[None, :])
    different_source = source_codes[:, None].ne(source_codes[None, :])
    positive_mask = same_identity & same_part & different_source
    negative_mask = ~same_identity
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    hardest_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
    hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
    return (scale * F.softplus((hardest_positive[valid] - hardest_negative[valid]) / scale)).mean()


def source_aware_dense_chamfer_contrastive(
    local_sets: torch.Tensor,
    labels: torch.Tensor,
    parts: torch.Tensor,
    source_codes: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-positive InfoNCE over symmetric local-set Chamfer similarity."""
    if local_sets.ndim != 3:
        raise ValueError(f"Expected [B, K, D] local sets, got {local_sets.shape}")
    if temperature <= 0.0:
        raise ValueError("Dense correspondence temperature must be positive")
    local_sets = F.normalize(local_sets.float(), dim=-1)
    patch_similarity = torch.einsum(
        "bkd,cld->bckl", local_sets, local_sets
    )
    query_to_target = patch_similarity.amax(dim=3).mean(dim=2)
    target_to_query = patch_similarity.amax(dim=2).mean(dim=2)
    pair_similarity = 0.5 * (query_to_target + target_to_query)

    same_identity = labels[:, None].eq(labels[None, :])
    same_part = parts[:, None].eq(parts[None, :])
    different_source = source_codes[:, None].ne(source_codes[None, :])
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = same_identity & same_part & different_source & ~self_mask
    negative_mask = ~same_identity & same_part
    candidate_mask = positive_mask | negative_mask
    valid_anchor = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    valid_count = valid_anchor.sum()
    if not valid_anchor.any():
        return local_sets.sum() * 0.0, valid_count

    logits = pair_similarity / temperature
    negative_infinity = torch.finfo(logits.dtype).min
    log_denominator = torch.logsumexp(
        logits.masked_fill(~candidate_mask, negative_infinity), dim=1
    )
    log_positive = torch.logsumexp(
        logits.masked_fill(~positive_mask, negative_infinity), dim=1
    )
    loss = (log_denominator - log_positive)[valid_anchor].mean()
    return loss, valid_count
