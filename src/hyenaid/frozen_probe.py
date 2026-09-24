from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
PARTS = ("head", "left_body", "right_body")
MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = max(1, min(size, int(round(width * scale))))
    resized_height = max(1, min(size, int(round(height * scale))))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC
    )
    fill = np.round(IMAGENET_MEAN * 255).astype(np.uint8)
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[...] = fill
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


class ProbeDataset(Dataset[tuple[torch.Tensor, int, int, int]]):
    def __init__(self, frame: pd.DataFrame, root: Path, image_size: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        row = self.frame.iloc[index]
        path = self.root / str(row.image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = letterbox(image, self.image_size).astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return (
            tensor,
            int(row.sample_index),
            int(row.label_index),
            int(row.part_index),
        )


@torch.inference_mode()
def encode_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    local_grid: int,
    tta_flip: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    def one_view(view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = model.forward_features(view)
        prefix_count = int(model.num_prefix_tokens)
        cls = F.normalize(tokens[:, 0].float(), dim=-1)
        patches = tokens[:, prefix_count:].float()
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise AssertionError(f"Patch count is not square: {patches.shape}")
        patch_grid = patches.reshape(
            patches.shape[0], side, side, patches.shape[-1]
        )
        mean_patch = F.normalize(patch_grid.mean(dim=(1, 2)), dim=-1)
        global_descriptor = F.normalize(torch.cat([cls, mean_patch], dim=-1), dim=-1)
        pooled = F.adaptive_avg_pool2d(
            patch_grid.permute(0, 3, 1, 2), (local_grid, local_grid)
        ).permute(0, 2, 3, 1)
        local_descriptor = F.normalize(pooled, dim=-1)
        return global_descriptor, local_descriptor

    global_descriptor, local_descriptor = one_view(images)
    if tta_flip:
        flip_global, flip_local = one_view(torch.flip(images, dims=(3,)))
        flip_local = torch.flip(flip_local, dims=(2,))
        global_descriptor = F.normalize(global_descriptor + flip_global, dim=-1)
        local_descriptor = F.normalize(local_descriptor + flip_local, dim=-1)
    return global_descriptor, local_descriptor.flatten(1, 2)


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    local_grid: int,
    tta_flip: bool,
) -> dict[str, torch.Tensor]:
    model.eval()
    outputs: dict[str, list[torch.Tensor]] = {
        "sample_index": [],
        "label_index": [],
        "part_index": [],
        "global": [],
        "local": [],
    }
    for step, (images, sample_indices, labels, parts) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            global_descriptor, local_descriptor = encode_batch(
                model, images, local_grid, tta_flip
            )
        outputs["sample_index"].append(sample_indices)
        outputs["label_index"].append(labels)
        outputs["part_index"].append(parts)
        outputs["global"].append(global_descriptor.half().cpu())
        outputs["local"].append(local_descriptor.half().cpu())
        if step % 25 == 0 or step == len(loader):
            print(f"extract {step}/{len(loader)}", flush=True)
    return {key: torch.cat(values, dim=0) for key, values in outputs.items()}


def class_reduce(
    similarities: torch.Tensor,
    gallery_labels: torch.Tensor,
    num_classes: int,
    mode: str,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.full(
        (similarities.shape[0], num_classes),
        -1.0,
        device=similarities.device,
        dtype=similarities.dtype,
    )
    available = torch.zeros(num_classes, dtype=torch.bool, device=similarities.device)
    for label in gallery_labels.unique(sorted=True):
        label_index = int(label)
        values = similarities[:, gallery_labels == label]
        if mode == "max":
            reduced = values.max(dim=1).values
        elif mode == "density":
            reduced = temperature * (
                torch.logsumexp(values / temperature, dim=1)
                - math.log(values.shape[1])
            )
        else:
            raise ValueError(mode)
        scores[:, label_index] = reduced
        available[label_index] = True
    return scores, available


def global_scores(
    train_global: torch.Tensor,
    train_labels: torch.Tensor,
    train_parts: torch.Tensor,
    query_global: torch.Tensor,
    query_parts: torch.Tensor,
    num_classes: int,
    mode: str,
) -> torch.Tensor:
    device = query_global.device
    all_similarities = query_global @ train_global.T
    fallback, _ = class_reduce(
        all_similarities, train_labels, num_classes, mode=mode
    )
    scores = fallback - 0.03
    for part in range(3):
        query_mask = query_parts == part
        gallery_mask = train_parts == part
        if not query_mask.any() or not gallery_mask.any():
            continue
        part_scores, available = class_reduce(
            query_global[query_mask] @ train_global[gallery_mask].T,
            train_labels[gallery_mask],
            num_classes,
            mode=mode,
        )
        selected = torch.where(query_mask)[0]
        scores[selected[:, None], torch.where(available)[0][None, :]] = part_scores[
            :, available
        ]
    return scores.to(device)


@torch.inference_mode()
def dynamic_local_scores(
    train_global: torch.Tensor,
    train_local: torch.Tensor,
    train_labels: torch.Tensor,
    train_parts: torch.Tensor,
    query_global: torch.Tensor,
    query_local: torch.Tensor,
    query_parts: torch.Tensor,
    num_classes: int,
    candidate_images: int,
    local_weight: float,
) -> torch.Tensor:
    scores = global_scores(
        train_global,
        train_labels,
        train_parts,
        query_global,
        query_parts,
        num_classes,
        mode="max",
    )
    keep_local = max(1, query_local.shape[1] // 2)
    for part in range(3):
        query_indices = torch.where(query_parts == part)[0]
        gallery_indices = torch.where(train_parts == part)[0]
        if not len(query_indices) or not len(gallery_indices):
            continue
        part_gallery_global = train_global[gallery_indices]
        for offset, query_index in enumerate(query_indices, start=1):
            image_similarities = query_global[query_index] @ part_gallery_global.T
            selected_count = min(candidate_images, len(gallery_indices))
            selected_local_indices = image_similarities.topk(selected_count).indices
            selected_gallery = gallery_indices[selected_local_indices]
            patch_similarities = torch.einsum(
                "ld,kmd->klm",
                query_local[query_index],
                train_local[selected_gallery],
            )
            query_to_gallery = patch_similarities.max(dim=2).values
            gallery_to_query = patch_similarities.max(dim=1).values
            local_similarity = 0.5 * (
                query_to_gallery.topk(keep_local, dim=1).values.mean(dim=1)
                + gallery_to_query.topk(keep_local, dim=1).values.mean(dim=1)
            )
            combined = (
                (1.0 - local_weight) * image_similarities[selected_local_indices]
                + local_weight * local_similarity
            )
            selected_labels = train_labels[selected_gallery]
            for label in selected_labels.unique(sorted=True):
                label_index = int(label)
                scores[query_index, label_index] = combined[
                    selected_labels == label
                ].max()
            if offset % 100 == 0 or offset == len(query_indices):
                print(
                    f"local part={PARTS[part]} {offset}/{len(query_indices)}",
                    flush=True,
                )
    return scores


def competition_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    per_part: dict[str, dict[str, float | int]] = {}
    for part in PARTS:
        subset = frame.loc[frame.part == part]
        true = subset.individual_id.astype(str)
        predicted = subset.predicted_id.astype(str)
        per_part[part] = {
            "macro_f1": float(
                f1_score(true, predicted, average="macro", zero_division=0)
            ),
            "top1_accuracy": float(accuracy_score(true, predicted)),
            "n_samples": int(len(subset)),
        }
    return {
        "final_score": float(
            np.mean([per_part[part]["macro_f1"] for part in PARTS])
        ),
        "per_part": per_part,
    }


def evaluate_scores(
    manifest: pd.DataFrame,
    valid_frame: pd.DataFrame,
    valid_sample_indices: torch.Tensor,
    scores: torch.Tensor,
    labels: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    predicted_indices = scores.argmax(dim=1).cpu().numpy()
    predictions = pd.DataFrame(
        {
            "sample_index": valid_sample_indices.cpu().numpy(),
            "predicted_id": [labels[index] for index in predicted_indices],
        }
    )
    result = valid_frame.merge(predictions, on="sample_index", validate="one_to_one")
    if len(result) != len(valid_frame):
        raise AssertionError("Validation prediction join changed row count")
    return competition_metrics(result), result


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--local-grid", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--candidate-images", type=int, default=48)
    parser.add_argument("--local-weight", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--no-tta-flip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fold not in range(5):
        raise ValueError("fold must be in 0..4")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    manifest = pd.read_csv(args.manifest).sort_values("sample_index").reset_index(drop=True)
    if len(manifest) != 4067 or manifest.fold.nunique() != 5:
        raise AssertionError("Unexpected manifest")
    if manifest.image_path.str.startswith("test/").any():
        raise AssertionError("Anonymous test image in manifest")
    train_frame = manifest.loc[manifest.fold != args.fold].copy()
    valid_frame = manifest.loc[manifest.fold == args.fold].copy()
    overlap = set(train_frame.source_group).intersection(valid_frame.source_group)
    if overlap:
        raise AssertionError(f"Source leakage: {len(overlap)} groups")
    labels = (
        manifest[["individual_id", "label_index"]]
        .drop_duplicates()
        .sort_values("label_index")
        .individual_id.astype(str)
        .tolist()
    )
    if len(labels) != 255:
        raise AssertionError(f"Expected 255 labels, got {len(labels)}")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(args.output_dir.parents[1] / ".cache/huggingface"))
    import timm

    cache_path = args.output_dir / (
        f"features_{MODEL_NAME.replace('.', '_')}_{args.image_size}_"
        f"grid{args.local_grid}_flip{int(not args.no_tta_flip)}.pt"
    )
    if cache_path.is_file():
        print(f"loading feature cache {cache_path}", flush=True)
        features = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        loader = DataLoader(
            ProbeDataset(manifest, args.competition_root, args.image_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        print(f"loading {MODEL_NAME}", flush=True)
        model = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=0,
            img_size=args.image_size,
        ).cuda().eval()
        started = time.time()
        features = extract_features(
            model,
            loader,
            torch.device("cuda"),
            args.local_grid,
            not args.no_tta_flip,
        )
        features["metadata"] = {
            "model": MODEL_NAME,
            "image_size": args.image_size,
            "local_grid": args.local_grid,
            "tta_flip": not args.no_tta_flip,
            "manifest_sha256": sha256_file(args.manifest),
            "elapsed_seconds": time.time() - started,
            "torch_version": torch.__version__,
            "timm_version": timm.__version__,
        }
        torch.save(features, cache_path)
        del model
        torch.cuda.empty_cache()

    sample_to_position = {
        int(sample_index): position
        for position, sample_index in enumerate(features["sample_index"].tolist())
    }
    train_positions = torch.tensor(
        [sample_to_position[int(value)] for value in train_frame.sample_index]
    )
    valid_positions = torch.tensor(
        [sample_to_position[int(value)] for value in valid_frame.sample_index]
    )
    device = torch.device("cuda")
    train_global = F.normalize(features["global"][train_positions].float(), dim=-1).to(device)
    train_local = F.normalize(features["local"][train_positions].float(), dim=-1).to(device)
    train_labels = features["label_index"][train_positions].long().to(device)
    train_parts = features["part_index"][train_positions].long().to(device)
    query_global = F.normalize(features["global"][valid_positions].float(), dim=-1).to(device)
    query_local = F.normalize(features["local"][valid_positions].float(), dim=-1).to(device)
    query_parts = features["part_index"][valid_positions].long().to(device)
    valid_sample_indices = features["sample_index"][valid_positions]

    results: dict[str, Any] = {"config": vars(args), "feature_metadata": features["metadata"]}
    results["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in results["config"].items()
    }
    for mode in ("max", "density"):
        scores = global_scores(
            train_global,
            train_labels,
            train_parts,
            query_global,
            query_parts,
            len(labels),
            mode,
        )
        metrics, prediction_frame = evaluate_scores(
            manifest, valid_frame, valid_sample_indices, scores, labels
        )
        name = f"global_{mode}"
        results[name] = metrics
        prediction_frame.to_csv(args.output_dir / f"fold_{args.fold}_{name}.csv", index=False)
        print(name, json.dumps(metrics, ensure_ascii=False), flush=True)

    local_scores = dynamic_local_scores(
        train_global,
        train_local,
        train_labels,
        train_parts,
        query_global,
        query_local,
        query_parts,
        len(labels),
        args.candidate_images,
        args.local_weight,
    )
    local_metrics, local_frame = evaluate_scores(
        manifest, valid_frame, valid_sample_indices, local_scores, labels
    )
    results["global_local_dynamic"] = local_metrics
    local_frame.to_csv(
        args.output_dir / f"fold_{args.fold}_global_local_dynamic.csv", index=False
    )
    print("global_local_dynamic", json.dumps(local_metrics, ensure_ascii=False), flush=True)
    write_json(results, args.output_dir / f"fold_{args.fold}_probe_metrics.json")


if __name__ == "__main__":
    main()

