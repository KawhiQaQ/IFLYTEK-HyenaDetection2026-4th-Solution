from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

from data import make_eval_loader
from engine import competition_metrics, prediction_frame, set_seed, write_json
from model import build_model


VERSION = "V2.54"
EXPECTED_MANIFEST_SHA256 = (
    "0470a4bb4630061ef06e92c50a74cb15ba457e476714294872ec426239a1ee6f"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "c8baedfc805feccd2310f2ca7420f806b9cc56763df3f06c00f998d1e061b3fc"
)
EXPECTED_BASELINE_SCORE = 0.6804990923313938
LOCAL_GRID = 7
PRIORITY_IMAGES = 40
GEOMETRIC_INLIERS = 12
TRANSLATION_TOLERANCE = 1
CLASSIFIER_WEIGHT = 0.5
RETRIEVAL_WEIGHT = 0.5
PART_NAMES = ("head", "left_body", "right_body")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def feature_metadata(
    frame: pd.DataFrame, features: dict[str, torch.Tensor]
) -> pd.DataFrame:
    if frame.sample_index.duplicated().any():
        raise AssertionError("Manifest sample_index is not unique")
    order = features["sample_index"].long().numpy()
    indexed = frame.set_index("sample_index", drop=False)
    if not set(order).issubset(set(indexed.index)):
        raise AssertionError("Feature extraction returned an unknown sample")
    metadata = indexed.loc[order].reset_index(drop=True)
    if not np.array_equal(metadata.sample_index.to_numpy(), order):
        raise AssertionError("Feature and metadata orders differ")
    return metadata


@torch.inference_mode()
def extract_dense_features(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """One model pass; a passive hook exports its trained covariance tokens."""
    model.eval()
    output: dict[str, list[torch.Tensor]] = {
        "sample_index": [],
        "part_index": [],
        "part_embedding": [],
        "classifier_score": [],
        "local": [],
    }
    captured: list[torch.Tensor] = []

    def capture_reduced(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        value: torch.Tensor,
    ) -> None:
        if captured:
            raise AssertionError("Covariance reduction ran more than once")
        captured.append(value.detach())

    reduction = getattr(model, "covariance_reduction", None)
    if reduction is None:
        raise AssertionError("V2.51 covariance reduction is unavailable")
    hook = reduction.register_forward_hook(capture_reduced)
    try:
        for images, _labels, parts, sample_indices, _sources in loader:
            captured.clear()
            images = images.to(device, non_blocking=True)
            part_device = parts.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                shared, part_embedding, _ = model.encode(
                    images,
                    part_device,
                    return_local=False,
                )
                classifier = model.inference_scores(
                    shared, part_embedding, part_device
                )
            if len(captured) != 1:
                raise AssertionError(
                    f"Expected one covariance capture, found {len(captured)}"
                )
            reduced = captured.pop().float()
            side = math.isqrt(reduced.shape[1])
            if side * side != reduced.shape[1] or reduced.shape[-1] != 64:
                raise AssertionError(
                    f"Unexpected reduced patch tensor {tuple(reduced.shape)}"
                )
            dense = reduced.transpose(1, 2).reshape(
                len(reduced), reduced.shape[-1], side, side
            )
            dense = F.adaptive_avg_pool2d(
                dense, (LOCAL_GRID, LOCAL_GRID)
            ).flatten(2).transpose(1, 2)
            dense = F.normalize(dense, dim=-1)
            output["sample_index"].append(sample_indices.cpu())
            output["part_index"].append(parts.cpu())
            output["part_embedding"].append(part_embedding.float().cpu())
            output["classifier_score"].append(classifier.float().cpu())
            output["local"].append(dense.half().cpu())
    finally:
        hook.remove()
    return {key: torch.cat(values, dim=0) for key, values in output.items()}


def geometric_local_scores(
    query: torch.Tensor,
    gallery: torch.Tensor,
) -> torch.Tensor:
    """Score [Q,B] pairs by mutual matches with translation consensus."""
    if query.ndim != 3 or gallery.ndim != 4:
        raise ValueError("Expected query [Q,L,D] and gallery [Q,B,L,D]")
    if query.shape[0] != gallery.shape[0] or query.shape[1:] != gallery.shape[2:]:
        raise ValueError("Query/gallery local descriptor shapes differ")
    q_count, shortlist, tokens, _dim = gallery.shape
    if tokens != LOCAL_GRID * LOCAL_GRID:
        raise AssertionError("Local token grid changed")
    similarity = torch.einsum(
        "qld,qbmd->qblm", query.float(), gallery.float()
    )
    q_best_value, q_best_gallery = similarity.max(dim=3)
    gallery_best_query = similarity.max(dim=2).indices
    query_indices = torch.arange(tokens, device=query.device).view(1, 1, -1)
    mutual = gallery_best_query.gather(2, q_best_gallery).eq(query_indices)

    axis = torch.arange(LOCAL_GRID, device=query.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((yy.flatten(), xx.flatten()), dim=1)
    query_coordinates = coordinates.view(1, 1, tokens, 2)
    matched_gallery_coordinates = coordinates[q_best_gallery]
    bins_per_axis = 2 * LOCAL_GRID - 1
    candidate_scores: list[torch.Tensor] = []
    for reflect in (False, True):
        aligned = matched_gallery_coordinates.clone()
        if reflect:
            aligned[..., 1] = LOCAL_GRID - 1 - aligned[..., 1]
        displacement = aligned - query_coordinates
        bin_index = (
            (displacement[..., 0] + LOCAL_GRID - 1) * bins_per_axis
            + displacement[..., 1]
            + LOCAL_GRID
            - 1
        ).long()
        support = torch.zeros(
            q_count,
            shortlist,
            bins_per_axis * bins_per_axis,
            dtype=q_best_value.dtype,
            device=query.device,
        )
        weights = q_best_value.clamp_min(0.0) * mutual
        support.scatter_add_(2, bin_index, weights)
        best_bin = support.argmax(dim=2)
        best_displacement = torch.stack(
            (
                best_bin.div(bins_per_axis, rounding_mode="floor")
                - LOCAL_GRID
                + 1,
                best_bin.remainder(bins_per_axis) - LOCAL_GRID + 1,
            ),
            dim=-1,
        ).unsqueeze(2)
        inlier = (
            (displacement - best_displacement).abs().amax(dim=-1)
            <= TRANSLATION_TOLERANCE
        )
        values = q_best_value.clamp_min(0.0) * mutual * inlier
        score = values.topk(GEOMETRIC_INLIERS, dim=2).values.sum(dim=2)
        candidate_scores.append(score / GEOMETRIC_INLIERS)
    return torch.maximum(candidate_scores[0], candidate_scores[1])


@dataclass
class StrictIsotonic:
    model: IsotonicRegression

    @classmethod
    def fit(cls, scores: np.ndarray, targets: np.ndarray) -> "StrictIsotonic":
        if len(np.unique(targets)) != 2:
            raise AssertionError("Calibration pairs need both positive and negative labels")
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(scores.astype(np.float64), targets.astype(np.float64))
        return cls(model=model)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        raw = scores.astype(np.float64)
        calibrated = self.model.predict(raw.reshape(-1)).reshape(raw.shape)
        return calibrated + np.finfo(np.float64).eps * raw

    def artifact(self) -> dict[str, list[float]]:
        return {
            "x_thresholds": self.model.X_thresholds_.astype(float).tolist(),
            "y_thresholds": self.model.y_thresholds_.astype(float).tolist(),
        }


def fit_train_only_calibration(
    train_features: dict[str, torch.Tensor],
    train_metadata: pd.DataFrame,
    device: torch.device,
) -> tuple[dict[int, tuple[StrictIsotonic, StrictIsotonic]], dict[str, Any]]:
    calibrators: dict[int, tuple[StrictIsotonic, StrictIsotonic]] = {}
    artifact: dict[str, Any] = {
        "scope": "fold-train only; same-part; different-source priority pairs",
        "local_grid": LOCAL_GRID,
        "priority_images": PRIORITY_IMAGES,
        "geometric_inliers": GEOMETRIC_INLIERS,
        "translation_tolerance": TRANSLATION_TOLERANCE,
        "parts": {},
    }
    for part_index, part_name in enumerate(PART_NAMES):
        feature_rows = torch.where(
            train_features["part_index"].long().eq(part_index)
        )[0]
        embeddings = F.normalize(
            train_features["part_embedding"][feature_rows].float(), dim=-1
        ).to(device)
        local = F.normalize(
            train_features["local"][feature_rows].float(), dim=-1
        ).to(device)
        labels = train_metadata.iloc[feature_rows.numpy()].label_index.to_numpy(
            dtype=np.int64
        )
        source_codes = pd.factorize(
            train_metadata.iloc[feature_rows.numpy()].source_group.astype(str),
            sort=True,
        )[0]
        source_tensor = torch.as_tensor(source_codes, device=device)
        global_matrix = embeddings @ embeddings.T
        same_source = source_tensor[:, None].eq(source_tensor[None, :])
        global_matrix.masked_fill_(same_source, -torch.inf)
        if embeddings.shape[0] <= PRIORITY_IMAGES:
            raise AssertionError(f"Too few {part_name} train images")
        global_values, gallery_indices = global_matrix.topk(
            PRIORITY_IMAGES, dim=1
        )
        if not torch.isfinite(global_values).all():
            raise AssertionError("Source exclusion left an invalid priority pair")

        local_values: list[torch.Tensor] = []
        chunk_size = 24
        for start in range(0, len(embeddings), chunk_size):
            stop = min(len(embeddings), start + chunk_size)
            selected = gallery_indices[start:stop]
            local_values.append(
                geometric_local_scores(
                    local[start:stop], local[selected]
                ).cpu()
            )
        local_matrix = torch.cat(local_values, dim=0)
        gallery_labels = labels[gallery_indices.cpu().numpy()]
        targets = (gallery_labels == labels[:, None]).reshape(-1)
        global_raw = global_values.float().cpu().numpy().reshape(-1)
        local_raw = local_matrix.float().numpy().reshape(-1)
        if int(targets.sum()) < 20:
            raise AssertionError(f"Too few positive calibration pairs for {part_name}")
        if np.any(source_codes[:, None] == source_codes[gallery_indices.cpu().numpy()]):
            raise AssertionError("Same-source calibration pair survived masking")
        global_calibrator = StrictIsotonic.fit(global_raw, targets)
        local_calibrator = StrictIsotonic.fit(local_raw, targets)
        calibrators[part_index] = (global_calibrator, local_calibrator)
        artifact["parts"][part_name] = {
            "train_images": int(len(embeddings)),
            "pairs": int(len(targets)),
            "positive_pairs": int(targets.sum()),
            "same_source_pairs": 0,
            "global_positive_mean": float(global_raw[targets].mean()),
            "global_negative_mean": float(global_raw[~targets].mean()),
            "local_positive_mean": float(local_raw[targets].mean()),
            "local_negative_mean": float(local_raw[~targets].mean()),
            "global": global_calibrator.artifact(),
            "local": local_calibrator.artifact(),
        }
    return calibrators, artifact


def percentile_rank(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("Percentile rank expects one dimension")
    if len(values) == 1:
        return np.ones(1, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


@torch.inference_mode()
def decode_queries(
    train_features: dict[str, torch.Tensor],
    train_metadata: pd.DataFrame,
    query_features: dict[str, torch.Tensor],
    calibrators: dict[int, tuple[StrictIsotonic, StrictIsotonic]],
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode without accepting or inspecting any query identity label."""
    final_scores = np.zeros((len(query_features["part_index"]), num_classes))
    retrieval_diagnostic = np.zeros_like(final_scores)
    for part_index, _part_name in enumerate(PART_NAMES):
        train_rows = torch.where(
            train_features["part_index"].long().eq(part_index)
        )[0]
        query_rows = torch.where(
            query_features["part_index"].long().eq(part_index)
        )[0]
        if not len(query_rows):
            continue
        train_embeddings = F.normalize(
            train_features["part_embedding"][train_rows].float(), dim=-1
        ).to(device)
        query_embeddings = F.normalize(
            query_features["part_embedding"][query_rows].float(), dim=-1
        ).to(device)
        train_local = F.normalize(
            train_features["local"][train_rows].float(), dim=-1
        ).to(device)
        query_local = F.normalize(
            query_features["local"][query_rows].float(), dim=-1
        ).to(device)
        train_labels = torch.as_tensor(
            train_metadata.iloc[train_rows.numpy()].label_index.to_numpy(
                dtype=np.int64, copy=True
            ),
            device=device,
        )
        available = np.zeros(num_classes, dtype=bool)
        available[np.unique(train_labels.cpu().numpy())] = True
        global_matrix = query_embeddings @ train_embeddings.T
        global_values, gallery_indices = global_matrix.topk(
            min(PRIORITY_IMAGES, len(train_embeddings)), dim=1
        )
        local_values: list[torch.Tensor] = []
        chunk_size = 24
        for start in range(0, len(query_embeddings), chunk_size):
            stop = min(len(query_embeddings), start + chunk_size)
            selected = gallery_indices[start:stop]
            local_values.append(
                geometric_local_scores(
                    query_local[start:stop], train_local[selected]
                ).cpu()
            )
        local_matrix = torch.cat(local_values, dim=0).numpy()
        global_calibrator, local_calibrator = calibrators[part_index]
        pair_global = global_calibrator.predict(global_values.cpu().numpy())
        pair_local = local_calibrator.predict(local_matrix)
        pair_fused = 0.5 * (pair_global + pair_local)
        selected_labels = train_labels[gallery_indices].cpu().numpy()

        class_global = torch.full(
            (len(query_embeddings), num_classes),
            -torch.inf,
            dtype=global_matrix.dtype,
            device=device,
        )
        class_global.scatter_reduce_(
            1,
            train_labels.view(1, -1).expand(len(query_embeddings), -1),
            global_matrix,
            reduce="amax",
            include_self=True,
        )
        class_global_np = class_global.cpu().numpy()
        classifier_np = query_features["classifier_score"][query_rows].numpy()
        for local_query_index, absolute_query_index in enumerate(query_rows.tolist()):
            retrieval = np.full(num_classes, -np.inf, dtype=np.float64)
            retrieval[available] = global_calibrator.predict(
                class_global_np[local_query_index, available]
            )
            shortlisted = np.full(num_classes, -np.inf, dtype=np.float64)
            np.maximum.at(
                shortlisted,
                selected_labels[local_query_index],
                pair_fused[local_query_index],
            )
            has_local = np.isfinite(shortlisted)
            retrieval[has_local] = shortlisted[has_local]

            classifier_rank = percentile_rank(classifier_np[local_query_index])
            retrieval_rank = classifier_rank.copy()
            retrieval_rank[available] = percentile_rank(retrieval[available])
            fused_rank = (
                CLASSIFIER_WEIGHT * classifier_rank
                + RETRIEVAL_WEIGHT * retrieval_rank
            )
            final_scores[absolute_query_index] = fused_rank
            retrieval_diagnostic[absolute_query_index] = retrieval_rank
    return (
        torch.from_numpy(final_scores).float(),
        torch.from_numpy(retrieval_diagnostic).float(),
    )


def build_loaded_model(
    checkpoint_path: Path,
    train_frame: pd.DataFrame,
    labels: list[str],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_model(
        config["model_variant"],
        num_classes=len(labels),
        image_size=int(config["image_size"]),
        embedding_dim=int(config["embedding_dim"]),
        local_queries=int(config["local_queries"]),
        pretrained=False,
        arc_scale=float(config["arc_scale"]),
        arc_margin=float(config["arc_margin"]),
        part_delta_scale=float(config["part_delta_scale"]),
        freeze_blocks=int(config["freeze_blocks"]),
        freeze_stages=int(config["freeze_stages"]),
        grad_checkpointing=True,
    )
    availability = torch.zeros(3, len(labels), dtype=torch.bool)
    availability[
        torch.as_tensor(train_frame.part_index.to_numpy(copy=True)),
        torch.as_tensor(train_frame.label_index.to_numpy(copy=True)),
    ] = True
    model.set_part_availability(availability)
    if hasattr(model, "set_part_counts"):
        counts = torch.zeros(3, len(labels), dtype=torch.float32)
        for (part, label), count in train_frame.groupby(
            ["part_index", "label_index"]
        ).size().items():
            counts[int(part), int(label)] = float(count)
        model.set_part_counts(counts)
    if hasattr(model, "set_class_counts"):
        model.set_class_counts(
            torch.bincount(
                torch.as_tensor(train_frame.label_index.to_numpy(copy=True)),
                minlength=len(labels),
            )
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, config, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fold != 0:
        raise ValueError("V2.54 is frozen for fold 0 only")
    manifest_hash = sha256_file(args.manifest)
    checkpoint_hash = sha256_file(args.checkpoint)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("Immutable fold manifest hash changed")
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError("V2.51 checkpoint hash changed")
    manifest = pd.read_csv(args.manifest)
    train_frame = manifest.loc[manifest.fold.ne(args.fold)].copy()
    valid_frame = manifest.loc[manifest.fold.eq(args.fold)].copy()
    source_overlap = set(train_frame.source_group.astype(str)).intersection(
        valid_frame.source_group.astype(str)
    )
    if source_overlap:
        raise AssertionError(f"Source leakage: {len(source_overlap)} groups")
    if train_frame.image_path.str.startswith("test/").any() or valid_frame.image_path.str.startswith("test/").any():
        raise AssertionError("Anonymous test image entered V2.54")
    labels = (
        manifest[["individual_id", "label_index"]]
        .drop_duplicates()
        .sort_values("label_index")
        .individual_id.astype(str)
        .tolist()
    )
    set_seed(20260719 + args.fold)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model, checkpoint_config, checkpoint = build_loaded_model(
        args.checkpoint, train_frame, labels, device
    )

    if args.smoke:
        smoke_frame = valid_frame.iloc[: args.eval_batch_size].copy()
        smoke_loader = make_eval_loader(
            smoke_frame,
            args.competition_root,
            int(checkpoint_config["image_size"]),
            args.eval_batch_size,
            args.workers,
            int(checkpoint_config["fold_seed"]),
            augmentation_profile=checkpoint_config["augmentation_profile"],
        )
        features = extract_dense_features(model, smoke_loader, device)
        if features["local"].shape != (len(smoke_frame), 49, 64):
            raise AssertionError("Dense local smoke shape changed")
        if not all(
            torch.isfinite(features[key]).all()
            for key in ("part_embedding", "classifier_score", "local")
        ):
            raise AssertionError("Dense local smoke produced non-finite values")
        result = {
            "version": VERSION,
            "batch": int(len(smoke_frame)),
            "local_shape": list(features["local"].shape),
            "source_overlap": 0,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {metrics_path}")
    frozen_config = {
        **serializable_args(args),
        "version": VERSION,
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_variant": checkpoint_config["model_variant"],
        "image_size": int(checkpoint_config["image_size"]),
        "non_tta": True,
        "local_grid": LOCAL_GRID,
        "priority_images": PRIORITY_IMAGES,
        "geometric_inliers": GEOMETRIC_INLIERS,
        "translation_tolerance": TRANSLATION_TOLERANCE,
        "classifier_weight": CLASSIFIER_WEIGHT,
        "retrieval_weight": RETRIEVAL_WEIGHT,
        "selection_decoder": "calibrated_global_local_rank_fusion",
        "calibration_scope": "fold-train only",
        "train_samples": int(len(train_frame)),
        "valid_samples": int(len(valid_frame)),
        "source_overlap": 0,
    }
    write_json(frozen_config, args.output_dir / "config.json")
    started = time.time()
    train_loader = make_eval_loader(
        train_frame,
        args.competition_root,
        int(checkpoint_config["image_size"]),
        args.eval_batch_size,
        args.workers,
        int(checkpoint_config["fold_seed"]),
        augmentation_profile=checkpoint_config["augmentation_profile"],
    )
    valid_loader = make_eval_loader(
        valid_frame,
        args.competition_root,
        int(checkpoint_config["image_size"]),
        args.eval_batch_size,
        args.workers,
        int(checkpoint_config["fold_seed"]),
        augmentation_profile=checkpoint_config["augmentation_profile"],
    )
    print("extracting fold-train features", flush=True)
    train_features = extract_dense_features(model, train_loader, device)
    print("extracting fold-valid features", flush=True)
    valid_features = extract_dense_features(model, valid_loader, device)
    train_metadata = feature_metadata(train_frame, train_features)
    valid_metadata = feature_metadata(valid_frame, valid_features)
    if set(train_metadata.sample_index) != set(train_frame.sample_index):
        raise AssertionError("Fold-train extraction coverage changed")
    if set(valid_metadata.sample_index) != set(valid_frame.sample_index):
        raise AssertionError("Fold-valid extraction coverage changed")
    torch.save(
        {
            "manifest_sha256": manifest_hash,
            "checkpoint_sha256": checkpoint_hash,
            "train": train_features,
            "valid": valid_features,
        },
        args.output_dir / "feature_cache.pt",
    )

    baseline_frame = prediction_frame(
        manifest, valid_features, valid_features["classifier_score"], labels
    )
    baseline_metrics = competition_metrics(baseline_frame)
    if abs(baseline_metrics["final_score"] - EXPECTED_BASELINE_SCORE) > 1e-12:
        raise AssertionError(
            f"V2.51 preflight score changed: {baseline_metrics['final_score']}"
        )
    incumbent_predictions = args.checkpoint.parent / "best_val_predictions.csv"
    if incumbent_predictions.is_file():
        expected = pd.read_csv(incumbent_predictions)[
            ["sample_index", "predicted_id"]
        ].sort_values("sample_index")
        actual = baseline_frame[
            ["sample_index", "predicted_id"]
        ].sort_values("sample_index")
        if not expected.reset_index(drop=True).equals(actual.reset_index(drop=True)):
            raise AssertionError("Passive dense hook changed V2.51 predictions")
    print(
        f"baseline preflight={baseline_metrics['final_score']:.12f}; "
        "fitting train-only calibration",
        flush=True,
    )
    calibrators, calibration_artifact = fit_train_only_calibration(
        train_features, train_metadata, device
    )
    write_json(calibration_artifact, args.output_dir / "calibration.json")

    query_decoder_features = {
        key: valid_features[key]
        for key in (
            "sample_index",
            "part_index",
            "part_embedding",
            "classifier_score",
            "local",
        )
    }
    fused_scores, retrieval_scores = decode_queries(
        train_features,
        train_metadata,
        query_decoder_features,
        calibrators,
        len(labels),
        device,
    )
    fused_frame = prediction_frame(manifest, valid_features, fused_scores, labels)
    retrieval_frame = prediction_frame(
        manifest, valid_features, retrieval_scores, labels
    )
    fused_metrics = competition_metrics(fused_frame)
    retrieval_metrics = competition_metrics(retrieval_frame)
    fused_frame = fused_frame.rename(
        columns={
            "predicted_id": "predicted_id",
            "confidence": "fused_rank_score",
        }
    )
    fused_frame = fused_frame.merge(
        baseline_frame[["sample_index", "predicted_id"]].rename(
            columns={"predicted_id": "classifier_predicted_id"}
        ),
        on="sample_index",
        validate="one_to_one",
    ).merge(
        retrieval_frame[["sample_index", "predicted_id"]].rename(
            columns={"predicted_id": "retrieval_predicted_id"}
        ),
        on="sample_index",
        validate="one_to_one",
    )
    predictions_path = args.output_dir / "val_predictions.csv"
    fused_frame.to_csv(predictions_path, index=False, encoding="utf-8")
    result = {
        "version": VERSION,
        "fold": args.fold,
        "decoder": "calibrated_global_local_rank_fusion",
        "elapsed_seconds": time.time() - started,
        "baseline_classifier_metrics": baseline_metrics,
        "retrieval_only_diagnostic_metrics": retrieval_metrics,
        "final_metrics": fused_metrics,
        "success_gate": 0.70,
        "success": bool(fused_metrics["final_score"] >= 0.70),
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    write_json(result, metrics_path)
    audit = {
        "rows": int(len(fused_frame)),
        "expected_rows": int(len(valid_frame)),
        "unique_sample_index": int(fused_frame.sample_index.nunique()),
        "missing": int(
            len(set(valid_frame.sample_index).difference(fused_frame.sample_index))
        ),
        "extra": int(
            len(set(fused_frame.sample_index).difference(valid_frame.sample_index))
        ),
        "illegal_predictions": int((~fused_frame.predicted_id.astype(str).isin(labels)).sum()),
        "null_predictions": int(fused_frame.predicted_id.isna().sum()),
        "source_overlap": 0,
        "anonymous_test_rows_used": 0,
        "calibration_train_samples": int(len(train_metadata)),
        "calibration_valid_samples": 0,
        "recomputed_final_score": fused_metrics["final_score"],
        "recomputed_part_macro_f1": {
            part: fused_metrics["per_part"][part]["macro_f1"]
            for part in PART_NAMES
        },
    }
    write_json(audit, args.output_dir / "audit.json")
    summary = {
        **result,
        "hashes": {
            "config": sha256_file(args.output_dir / "config.json"),
            "calibration": sha256_file(args.output_dir / "calibration.json"),
            "predictions": sha256_file(predictions_path),
            "metrics": sha256_file(metrics_path),
            "audit": sha256_file(args.output_dir / "audit.json"),
        },
    }
    write_json(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
