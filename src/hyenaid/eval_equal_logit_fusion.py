from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from data import build_crop_geometry_stats, make_eval_loader
from engine import competition_metrics, extract, prediction_frame, set_seed, write_json
from model import build_model


EXPECTED_MANIFEST_SHA256 = (
    "0470a4bb4630061ef06e92c50a74cb15ba457e476714294872ec426239a1ee6f"
)
EXPECTED_MODEL_VARIANT = "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2"
EXPECTED_P8_SHA256 = (
    "4c39ce6746121d180f81c0afb4d46530570e8cd3bb0b1e05c7ac219f1ea338aa"
)
EXPECTED_V61_SHA256 = (
    "42ac52b0912c84d9fea1bcbe3558835fb084fc4e852fbdb4d21a18b6987d4787"
)
EXPECTED_P8_SCORE = 0.6855191601978851
EXPECTED_V61_SCORE = 0.6834445454012879
USEFUL_GATE = EXPECTED_P8_SCORE + 0.004
FOLD = 0
WEIGHT_P8 = 0.5
WEIGHT_V61 = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preregistered fold-0 V2.63-P8 + V2.61 equal-logit gate"
    )
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-p8", type=Path, required=True)
    parser.add_argument("--checkpoint-v61", type=Path, required=True)
    parser.add_argument("--reference-p8", type=Path, required=True)
    parser.add_argument("--reference-v61", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    return parser.parse_args()


def label_order(manifest: pd.DataFrame) -> list[str]:
    rows = (
        manifest[["individual_id", "label_index"]]
        .drop_duplicates()
        .sort_values("label_index")
    )
    if rows.label_index.astype(int).tolist() != list(range(len(rows))):
        raise AssertionError("Manifest label indices are not contiguous")
    return rows.individual_id.astype(str).tolist()


def set_fold_train_buffers(
    model: torch.nn.Module,
    train: pd.DataFrame,
    num_classes: int,
) -> None:
    availability = torch.zeros(3, num_classes, dtype=torch.bool)
    availability[
        torch.as_tensor(train.part_index.to_numpy(copy=True)),
        torch.as_tensor(train.label_index.to_numpy(copy=True)),
    ] = True
    model.set_part_availability(availability)
    if hasattr(model, "set_part_counts"):
        counts = torch.zeros(3, num_classes, dtype=torch.float32)
        for (part, label), count in train.groupby(
            ["part_index", "label_index"]
        ).size().items():
            counts[int(part), int(label)] = float(count)
        model.set_part_counts(counts)
    if hasattr(model, "set_class_counts"):
        model.set_class_counts(
            torch.bincount(
                torch.as_tensor(train.label_index.to_numpy(copy=True)),
                minlength=num_classes,
            )
        )


def assert_reference_predictions(
    frame: pd.DataFrame,
    reference_path: Path,
    component: str,
) -> None:
    reference = pd.read_csv(reference_path)
    required = {"sample_index", "predicted_id"}
    if not required.issubset(reference.columns):
        raise AssertionError(f"{component} reference prediction columns changed")
    expected = reference[["sample_index", "predicted_id"]].copy()
    actual = frame[["sample_index", "predicted_id"]].copy()
    expected["predicted_id"] = expected.predicted_id.astype(str)
    actual["predicted_id"] = actual.predicted_id.astype(str)
    expected = expected.sort_values("sample_index").reset_index(drop=True)
    actual = actual.sort_values("sample_index").reset_index(drop=True)
    if not expected.equals(actual):
        disagreements = int(
            (expected.predicted_id != actual.predicted_id).sum()
        ) if len(expected) == len(actual) else -1
        raise AssertionError(
            f"{component} checkpoint did not reproduce its stored non-TTA "
            f"predictions; disagreements={disagreements}"
        )


def extract_component(
    *,
    name: str,
    checkpoint_path: Path,
    expected_sha256: str,
    expected_score: float,
    reference_path: Path,
    manifest: pd.DataFrame,
    train: pd.DataFrame,
    valid_loader: Any,
    labels: list[str],
    geometry_stats: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise AssertionError(
            f"{name} checkpoint SHA-256 {actual_sha256} != {expected_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if (
        checkpoint.get("labels") != labels
        or int(checkpoint.get("epoch", -999)) < 0
        or config.get("model_variant") != EXPECTED_MODEL_VARIANT
        or int(config.get("fold", -1)) != FOLD
        or config.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or config.get("init_checkpoint") is not None
        or config.get("init_checkpoint_sha256") is not None
        or int(config.get("train_samples", -1)) != 3253
        or int(config.get("valid_samples", -1)) != 814
        or int(config.get("image_size", -1)) != 448
        or config.get("augmentation_profile") != "arbase_source_jitter"
        or config.get("released_crop_geometry_stats") != geometry_stats
    ):
        raise AssertionError(f"{name} violates the fixed-fold from-generic contract")
    stored_score = float(checkpoint["classifier_metrics"]["final_score"])
    if abs(stored_score - expected_score) > 1e-10:
        raise AssertionError(
            f"{name} stored score {stored_score} != expected {expected_score}"
        )

    set_seed(int(config["fold_seed"]))
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
    set_fold_train_buffers(model, train, len(labels))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    features = extract(
        model,
        valid_loader,
        device,
        tta_flip=False,
        return_local=False,
        local_grid=int(config["local_grid"]),
    )
    frame = prediction_frame(
        manifest, features, features["classifier_score"], labels
    )
    metrics = competition_metrics(frame)
    if abs(float(metrics["final_score"]) - expected_score) > 1e-10:
        raise AssertionError(
            f"{name} re-extracted score {metrics['final_score']} != {expected_score}"
        )
    assert_reference_predictions(frame, reference_path, name)
    core = {
        key: features[key].clone()
        for key in ("sample_index", "label_index", "part_index", "classifier_score")
    }
    provenance = {
        "name": name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": actual_sha256,
        "checkpoint_epoch_zero_based": int(checkpoint["epoch"]),
        "stored_non_tta_score": stored_score,
        "reextracted_non_tta_score": float(metrics["final_score"]),
        "model_variant": config["model_variant"],
        "pretraining_source": config.get("pretraining_source"),
        "init_checkpoint": None,
    }
    del frame, features, model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return core, metrics, provenance


def main() -> None:
    args = parse_args()
    args.competition_root = args.competition_root.resolve()
    args.manifest = args.manifest.resolve()
    args.checkpoint_p8 = args.checkpoint_p8.resolve()
    args.checkpoint_v61 = args.checkpoint_v61.resolve()
    args.reference_p8 = args.reference_p8.resolve()
    args.reference_v61 = args.reference_v61.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {args.output_dir}")
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("Unexpected immutable manifest SHA-256")
    manifest = pd.read_csv(args.manifest)
    train = manifest.loc[manifest.fold != FOLD].copy()
    valid = manifest.loc[manifest.fold == FOLD].copy()
    if len(manifest) != 4067 or len(train) != 3253 or len(valid) != 814:
        raise AssertionError("Immutable fold row counts changed")
    if set(train.source_group).intersection(valid.source_group):
        raise AssertionError("Fold-0 train/validation source leakage")
    if manifest.image_path.astype(str).str.startswith("test/").any():
        raise AssertionError("Anonymous test row entered immutable manifest")
    labels = label_order(manifest)
    if len(labels) != 255:
        raise AssertionError("Expected exactly 255 legal training identities")
    geometry_stats = build_crop_geometry_stats(train, args.competition_root)

    valid_loader = make_eval_loader(
        valid,
        args.competition_root,
        448,
        args.eval_batch_size,
        args.workers,
        20260719,
        augmentation_profile="arbase_source_jitter",
        geometry_stats=geometry_stats,
    )
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the locked checkpoint extraction")
    p8, p8_metrics, p8_provenance = extract_component(
        name="V2.63-P8",
        checkpoint_path=args.checkpoint_p8,
        expected_sha256=EXPECTED_P8_SHA256,
        expected_score=EXPECTED_P8_SCORE,
        reference_path=args.reference_p8,
        manifest=manifest,
        train=train,
        valid_loader=valid_loader,
        labels=labels,
        geometry_stats=geometry_stats,
        device=device,
    )
    print(
        f"V2.63-P8 nonTTA={p8_metrics['final_score']:.10f}", flush=True
    )
    v61, v61_metrics, v61_provenance = extract_component(
        name="V2.61",
        checkpoint_path=args.checkpoint_v61,
        expected_sha256=EXPECTED_V61_SHA256,
        expected_score=EXPECTED_V61_SCORE,
        reference_path=args.reference_v61,
        manifest=manifest,
        train=train,
        valid_loader=valid_loader,
        labels=labels,
        geometry_stats=geometry_stats,
        device=device,
    )
    print(f"V2.61 nonTTA={v61_metrics['final_score']:.10f}", flush=True)

    for key in ("sample_index", "label_index", "part_index"):
        if not torch.equal(p8[key], v61[key]):
            raise AssertionError(f"Component alignment differs for {key}")
    if p8["classifier_score"].shape != (814, 255):
        raise AssertionError("Unexpected component score shape")
    if not torch.isfinite(p8["classifier_score"]).all() or not torch.isfinite(
        v61["classifier_score"]
    ).all():
        raise FloatingPointError("Component score matrix is non-finite")
    fused_scores = (
        WEIGHT_P8 * p8["classifier_score"]
        + WEIGHT_V61 * v61["classifier_score"]
    )
    fused_frame = prediction_frame(manifest, p8, fused_scores, labels)
    fused_metrics = competition_metrics(fused_frame)
    p8_frame = prediction_frame(
        manifest, p8, p8["classifier_score"], labels
    )
    v61_frame = prediction_frame(
        manifest, v61, v61["classifier_score"], labels
    )
    delta = float(fused_metrics["final_score"]) - EXPECTED_P8_SCORE
    passed_gate = float(fused_metrics["final_score"]) >= USEFUL_GATE

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_artifact = {
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "fold": FOLD,
        "tta_flip": False,
        "weights": {"V2.63-P8": WEIGHT_P8, "V2.61": WEIGHT_V61},
        "labels": labels,
        "sample_index": p8["sample_index"],
        "label_index": p8["label_index"],
        "part_index": p8["part_index"],
        "v2_63_p8_scores": p8["classifier_score"],
        "v2_61_scores": v61["classifier_score"],
        "components": [p8_provenance, v61_provenance],
    }
    torch.save(score_artifact, args.output_dir / "component_scores.pt")
    p8_frame.to_csv(
        args.output_dir / "v2_63_p8_predictions.csv", index=False, encoding="utf-8"
    )
    v61_frame.to_csv(
        args.output_dir / "v2_61_predictions.csv", index=False, encoding="utf-8"
    )
    fused_frame.to_csv(
        args.output_dir / "predictions.csv", index=False, encoding="utf-8"
    )
    result = {
        "experiment": "v2_63_p8_v2_61_equal_logit_fusion",
        "fold": FOLD,
        "decoder": "fixed_equal_logit_average",
        "weights": {"V2.63-P8": WEIGHT_P8, "V2.61": WEIGHT_V61},
        "tta_flip": False,
        "component_metrics": {
            "V2.63-P8": p8_metrics,
            "V2.61": v61_metrics,
        },
        "fused_metrics": fused_metrics,
        "reference_incumbent": EXPECTED_P8_SCORE,
        "delta_vs_incumbent": delta,
        "useful_gate": USEFUL_GATE,
        "passed_preregistered_gate": passed_gate,
        "checkpoint_provenance": [p8_provenance, v61_provenance],
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(result, args.output_dir / "metrics.json")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
