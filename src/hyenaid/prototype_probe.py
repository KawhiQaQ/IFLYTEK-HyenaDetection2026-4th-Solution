from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from data import make_eval_loader
from engine import competition_metrics, extract, prediction_frame, set_seed, write_json
from model import build_model


def class_centroids(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = F.normalize(embeddings.float(), dim=-1)
    centroids = torch.zeros(num_classes, embeddings.shape[1])
    available = torch.zeros(num_classes, dtype=torch.bool)
    for label in labels.unique(sorted=True):
        index = int(label)
        centroids[index] = F.normalize(
            embeddings[labels == label].mean(dim=0), dim=0
        )
        available[index] = True
    return centroids, available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.fold_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        args.fold_dir / "best.pt", map_location="cpu", weights_only=False
    )
    manifest = pd.read_csv(args.manifest)
    fold = int(config["fold"])
    train = manifest.loc[manifest.fold != fold].copy()
    valid = manifest.loc[manifest.fold == fold].copy()
    labels = checkpoint["labels"]
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
        grad_checkpointing=False,
    )
    model.load_state_dict(checkpoint["model"])
    availability = torch.zeros(3, len(labels), dtype=torch.bool)
    availability[
        torch.as_tensor(train.part_index.to_numpy(copy=True)),
        torch.as_tensor(train.label_index.to_numpy(copy=True)),
    ] = True
    model.set_part_availability(availability)
    device = torch.device("cuda")
    model.to(device)
    loader_kwargs = {
        "competition_root": args.competition_root,
        "image_size": int(config["image_size"]),
        "batch_size": int(config["eval_batch_size"]),
        "workers": int(config["workers"]),
        "seed": int(config["fold_seed"]),
        "augmentation_profile": config["augmentation_profile"],
    }
    train_features = extract(
        model,
        make_eval_loader(train, **loader_kwargs),
        device,
        tta_flip=False,
        return_local=False,
    )
    valid_features = extract(
        model,
        make_eval_loader(valid, **loader_kwargs),
        device,
        tta_flip=False,
        return_local=False,
    )

    train_embedding = train_features["shared_embedding"]
    valid_embedding = F.normalize(valid_features["shared_embedding"].float(), dim=-1)
    global_centroids, _ = class_centroids(
        train_embedding, train_features["label_index"], len(labels)
    )
    scores = valid_embedding @ global_centroids.T
    for part in range(3):
        train_mask = train_features["part_index"] == part
        query_mask = valid_features["part_index"] == part
        centroids, part_available = class_centroids(
            train_embedding[train_mask],
            train_features["label_index"][train_mask],
            len(labels),
        )
        query_indices = torch.where(query_mask)[0]
        class_indices = torch.where(part_available)[0]
        scores[query_indices[:, None], class_indices[None, :]] = (
            valid_embedding[query_mask] @ centroids[part_available].T
        )

    frame = prediction_frame(manifest, valid_features, scores, labels)
    metrics = competition_metrics(frame)
    frame.to_csv(args.fold_dir / "prototype_val_predictions.csv", index=False)
    write_json(metrics, args.fold_dir / "prototype_metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
