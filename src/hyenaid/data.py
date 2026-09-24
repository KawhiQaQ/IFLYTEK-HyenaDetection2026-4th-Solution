from __future__ import annotations

import math
import random
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
TARGET_PART_LABELS = {"head", "left body", "right body"}
GEOMETRY_FEATURE_NAMES = ("log_crop_pixels", "log_crop_aspect")
FOREGROUND_MODES = {
    "none",
    "sam_bg",
    "sam_aux",
    "sam_view",
    "sam_part_view",
}

cv2.setNumThreads(0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_foreground_mask_artifact(
    frame: pd.DataFrame,
    mask_root: Path,
    *,
    verify_mask_hashes: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate a SAM artifact against only the supplied training rows."""
    if frame.empty:
        raise AssertionError("Foreground masks require non-empty training rows")
    if frame.image_path.astype(str).str.startswith("test/").any():
        raise AssertionError("Anonymous test row entered foreground-mask validation")
    mask_root = mask_root.resolve()
    index_path = mask_root / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = pd.read_csv(index_path)
    required_columns = {
        "sample_index",
        "image_path",
        "source_group",
        "part",
        "part_index",
        "valid",
        "method",
        "candidate_index",
        "score",
        "area",
        "mask_sha256",
    }
    if set(index.columns) != required_columns:
        raise AssertionError(
            f"Unexpected foreground index columns: {list(index.columns)}"
        )
    if len(index) != len(frame) or index.sample_index.nunique() != len(index):
        raise AssertionError("Foreground index row geometry differs from training frame")
    expected = frame[
        ["sample_index", "image_path", "source_group", "part", "part_index"]
    ].copy()
    actual = index[
        ["sample_index", "image_path", "source_group", "part", "part_index"]
    ].copy()
    expected = expected.sort_values("sample_index").reset_index(drop=True)
    actual = actual.sort_values("sample_index").reset_index(drop=True)
    for column in ("image_path", "source_group", "part"):
        expected[column] = expected[column].astype(str)
        actual[column] = actual[column].astype(str)
    if not expected.equals(actual):
        raise AssertionError("Foreground index does not exactly match training rows")
    if index.image_path.astype(str).str.startswith("test/").any():
        raise AssertionError("Anonymous test path entered foreground index")
    valid = index.valid.astype(bool)
    valid_index = index.loc[valid]
    if valid_index.empty:
        raise AssertionError("Foreground artifact contains no valid mask")
    if not valid_index.score.between(0.85, 1.0).all():
        raise AssertionError("Foreground score gate changed")
    if not valid_index.area.between(0.10, 0.90).all():
        raise AssertionError("Foreground area gate changed")
    if not set(valid_index.method.astype(str)).issubset({"center", "box_center"}):
        raise AssertionError("Unexpected foreground prompt method")
    for row in index.itertuples(index=False):
        path = mask_root / "masks" / f"{int(row.sample_index):06d}.png"
        if bool(row.valid):
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_mask_hashes and sha256_file(path) != str(row.mask_sha256):
                raise AssertionError(f"Foreground mask hash mismatch: {path}")
        elif path.exists():
            raise AssertionError(f"Invalid foreground row has a mask file: {path}")
    by_part: dict[str, dict[str, float | int]] = {}
    for part, group in index.groupby("part", sort=True):
        count = int(group.valid.astype(bool).sum())
        fraction = count / len(group)
        if fraction < 0.70:
            raise AssertionError(
                f"Foreground validity gate failed for {part}: {fraction:.4f}"
            )
        by_part[str(part)] = {
            "rows": int(len(group)),
            "valid": count,
            "valid_fraction": float(fraction),
        }
    metadata: dict[str, object] = {
        "root": str(mask_root),
        "index_sha256": sha256_file(index_path),
        "rows": int(len(index)),
        "valid": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "by_part": by_part,
        "mask_hashes_verified": bool(verify_mask_hashes),
    }
    return index.sort_values("sample_index").reset_index(drop=True), metadata


def foreground_safe_background_blur(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    sigma_fraction: float,
) -> np.ndarray:
    """Blur only the background while exactly preserving teacher foreground."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HWC colour image")
    mask = np.asarray(foreground_mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("Foreground mask/image geometry differs")
    if not 0.015 <= sigma_fraction <= 0.035:
        raise ValueError("Foreground background-blur sigma differs from SPEC")
    short_side = min(image.shape[:2])
    sigma = sigma_fraction * short_side
    radius = max(1, math.ceil(0.01 * short_side))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    protected = cv2.dilate(np.uint8(mask), kernel).astype(np.float32)
    alpha = cv2.GaussianBlur(
        protected,
        (0, 0),
        sigmaX=max(0.5, 0.005 * short_side),
        sigmaY=max(0.5, 0.005 * short_side),
    )
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    output = np.rint(
        alpha * image.astype(np.float32)
        + (1.0 - alpha) * blurred.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    output[mask] = image[mask]
    if not np.array_equal(output[mask], image[mask]):
        raise AssertionError("Foreground pixels changed during background blur")
    return output


def quality_normalize_bgr(image: np.ndarray) -> np.ndarray:
    """Deterministic luminance-only local contrast and mild unsharp view."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Quality normalization expects uint8 BGR")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(luminance)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.25, blurred, -0.25, 0.0)
    return cv2.cvtColor(
        cv2.merge((sharpened, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )


def apply_quality_view_bgr(
    image: np.ndarray, part_index: int, *, body_only: bool
) -> np.ndarray:
    """Apply the fixed quality view, optionally only to organizer body rows."""
    if part_index not in {0, 1, 2}:
        raise ValueError(f"Unexpected organizer part index: {part_index}")
    if body_only and part_index == 0:
        return image
    return quality_normalize_bgr(image)


def build_transform(
    image_size: int, training: bool, profile: str = "default"
) -> A.Compose:
    if profile == "arbase_radio":
        # NVIDIA RADIO owns its published input conditioner and expects RGB
        # tensors in [0, 1].  Keep padding at the conditioner's CLIP mean so
        # padded pixels become zero after the model-side normalization.
        normalization_mean, normalization_std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        mean_fill = tuple(round(value * 255) for value in OPENAI_CLIP_MEAN)
    elif profile == "arbase_unit":
        # TIPS publishes a raw RGB [0,1] input contract without a model-side
        # conditioner. Use a neutral ImageNet-colour fill only for crop jitter
        # padding while preserving the exact unit-range tensor.
        normalization_mean, normalization_std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        mean_fill = tuple(round(value * 255) for value in IMAGENET_MEAN)
    elif profile == "arbase_clip":
        normalization_mean, normalization_std = OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
    elif profile == "arbase_siglip":
        normalization_mean, normalization_std = SIGLIP_MEAN, SIGLIP_STD
    else:
        normalization_mean, normalization_std = IMAGENET_MEAN, IMAGENET_STD
    if profile not in {"arbase_radio", "arbase_unit"}:
        mean_fill = tuple(round(value * 255) for value in normalization_mean)
    if profile in {
        "arbase",
        "arbase_radio",
        "arbase_unit",
        "arbase_clip",
        "arbase_siglip",
        "arbase_letterbox",
        "arbase_head",
        "degraded",
    }:
        if profile == "arbase_letterbox":
            # Preserve the geometry of already tight official crops.  Mean
            # padding becomes zero after ImageNet normalization, so elongated
            # crops are not stretched merely to satisfy ViT batching.
            operations: list[Any] = [
                A.LongestMaxSize(
                    max_size=image_size,
                    interpolation=cv2.INTER_CUBIC,
                )
            ]
        else:
            operations = [
                A.Resize(
                    height=image_size,
                    width=image_size,
                    interpolation=cv2.INTER_CUBIC,
                )
            ]
        if training:
            operations.extend(
                [
                    A.PadIfNeeded(
                        min_height=image_size + 20,
                        min_width=image_size + 20,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=mean_fill,
                        position="center",
                    ),
                    A.RandomCrop(height=image_size, width=image_size),
                    A.HorizontalFlip(p=0.5),
                ]
            )
            if profile == "degraded":
                # Preserve most clean examples while simulating the blur,
                # resampling, noise, compression and illumination variation
                # found in low-quality camera-trap crops.
                operations.extend(
                    [
                        A.OneOf(
                            [
                                A.RandomBrightnessContrast(
                                    brightness_limit=0.22,
                                    contrast_limit=0.22,
                                    p=1.0,
                                ),
                                A.RandomGamma(gamma_limit=(65, 145), p=1.0),
                                A.CLAHE(clip_limit=(1.2, 2.5), p=1.0),
                            ],
                            p=0.35,
                        ),
                        A.Sequential(
                            [
                                A.OneOf(
                                    [
                                        A.GaussianBlur(
                                            blur_limit=(3, 7),
                                            sigma_limit=(0.4, 2.2),
                                            p=1.0,
                                        ),
                                        A.MotionBlur(blur_limit=(3, 9), p=1.0),
                                        A.Defocus(
                                            radius=(2, 5),
                                            alias_blur=(0.1, 0.4),
                                            p=1.0,
                                        ),
                                        A.Downscale(
                                            scale_range=(0.30, 0.72),
                                            interpolation_pair={
                                                "downscale": cv2.INTER_AREA,
                                                "upscale": cv2.INTER_CUBIC,
                                            },
                                            p=1.0,
                                        ),
                                        A.Downscale(
                                            scale_range=(0.30, 0.72),
                                            interpolation_pair={
                                                "downscale": cv2.INTER_LINEAR,
                                                "upscale": cv2.INTER_LINEAR,
                                            },
                                            p=1.0,
                                        ),
                                        A.Downscale(
                                            scale_range=(0.30, 0.72),
                                            interpolation_pair={
                                                "downscale": cv2.INTER_NEAREST,
                                                "upscale": cv2.INTER_NEAREST,
                                            },
                                            p=1.0,
                                        ),
                                    ],
                                    p=1.0,
                                ),
                                A.GaussNoise(
                                    std_range=(0.008, 0.055),
                                    per_channel=True,
                                    p=0.70,
                                ),
                                A.ImageCompression(
                                    quality_range=(38, 92), p=0.80
                                ),
                            ],
                            p=0.35,
                        ),
                        A.ToGray(p=0.08),
                    ]
                )
            elif profile == "arbase_head":
                # Head crops have the lowest few-shot transfer and the widest
                # pose/illumination variation. These mild identity-preserving
                # transforms are applied only to head rows by HyenaDataset;
                # flank markings keep the exact ARBase pipeline.
                operations.extend(
                    [
                        A.Affine(
                            scale=(0.94, 1.06),
                            translate_percent=(0.0, 0.0),
                            rotate=(-7, 7),
                            shear=(-1.5, 1.5),
                            interpolation=cv2.INTER_CUBIC,
                            border_mode=cv2.BORDER_REFLECT_101,
                            p=0.60,
                        ),
                        A.ColorJitter(
                            brightness=0.12,
                            contrast=0.12,
                            saturation=0.08,
                            hue=0.015,
                            p=0.45,
                        ),
                    ]
                )
        elif profile == "arbase_letterbox":
            operations.append(
                A.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=mean_fill,
                    position="center",
                )
            )
        operations.extend(
            [
                A.Normalize(mean=normalization_mean, std=normalization_std),
                ToTensorV2(),
            ]
        )
        return A.Compose(operations)
    if profile != "default":
        raise ValueError(f"Unknown augmentation profile: {profile}")
    operations: list[Any] = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_CUBIC),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=mean_fill,
        ),
    ]
    if training:
        operations.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.92, 1.08),
                    translate_percent=(-0.035, 0.035),
                    rotate=(-8, 8),
                    shear=(-2.5, 2.5),
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.60,
                ),
                A.ColorJitter(
                    brightness=0.16,
                    contrast=0.16,
                    saturation=0.12,
                    hue=0.025,
                    p=0.50,
                ),
                A.ToGray(p=0.06),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
                        A.ImageCompression(quality_range=(72, 98), p=1.0),
                        A.CLAHE(clip_limit=(1.2, 2.0), p=1.0),
                    ],
                    p=0.18,
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 2),
                    hole_height_range=(0.03, 0.10),
                    hole_width_range=(0.03, 0.10),
                    fill=mean_fill,
                    p=0.08,
                ),
            ]
        )
    operations.extend(
        [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    )
    return A.Compose(operations)


def validate_model_augmentation_profile(
    model_variant: str, augmentation_profile: str
) -> None:
    if model_variant.startswith("radio_"):
        expected = "arbase_source_jitter_radio"
    elif model_variant.startswith("tips_"):
        expected = "arbase_source_jitter_tips"
    elif model_variant.startswith(("eva02_", "bioclip_", "bioclip2_")):
        expected = "arbase_clip"
    elif model_variant.startswith("siglip2_"):
        expected = "arbase_siglip"
    else:
        expected = None
    specialized_profiles = {
        "arbase_clip",
        "arbase_siglip",
        "arbase_source_jitter_radio",
        "arbase_source_jitter_tips",
    }
    if expected is not None and augmentation_profile != expected:
        raise ValueError(
            f"{model_variant} requires {expected}, got {augmentation_profile}"
        )
    if expected is None and augmentation_profile in specialized_profiles:
        raise ValueError(
            f"{model_variant} requires an ImageNet-normalized profile, got "
            f"{augmentation_profile}"
        )


def build_official_source_crop_lookup(
    frame: pd.DataFrame, competition_root: Path
) -> dict[str, tuple[str, tuple[int, int, int, int]]]:
    """Resolve active fold-train crops to their organizer XML boxes."""
    xml_dir = competition_root / "scripts" / "hyena_xml"
    lookup: dict[str, tuple[str, tuple[int, int, int, int]]] = {}
    for source_group in sorted(frame.source_group.astype(str).unique()):
        xml_path = xml_dir / f"{source_group}.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        root = ET.parse(xml_path).getroot()
        folder = (root.findtext("folder") or "").strip()
        filename = (root.findtext("filename") or "").strip()
        if not folder or not filename:
            raise ValueError(f"Missing folder/filename in {xml_path}")
        source_image = (
            Path("hyena") / "hyena_images" / folder / filename
        ).as_posix()
        label_counts: Counter[str] = Counter()
        for obj in root.findall("object"):
            label = (obj.findtext("name") or "").strip()
            if label not in TARGET_PART_LABELS:
                continue
            label_counts[label] += 1
            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"Missing bndbox for {label} in {xml_path}")
            coordinates = tuple(
                int((box.findtext(name) or "").strip())
                for name in ("xmin", "ymin", "xmax", "ymax")
            )
            stem = Path(filename).stem
            suffix = Path(filename).suffix.lower() or ".jpg"
            label_suffix = label.replace(" ", "_")
            occurrence_suffix = (
                "" if label_counts[label] == 1 else f"_{label_counts[label]}"
            )
            output_name = f"{stem}_{label_suffix}{occurrence_suffix}{suffix}"
            crop_path = (
                Path("hyena")
                / "cropped_images"
                / label
                / folder
                / output_name
            ).as_posix()
            if crop_path in lookup:
                raise AssertionError(f"Duplicate XML crop path: {crop_path}")
            lookup[crop_path] = (source_image, coordinates)

    active_paths = set(frame.image_path.astype(str))
    missing = sorted(active_paths.difference(lookup))
    if missing:
        raise AssertionError(
            f"Official XML box coverage missing {len(missing)} active crops: "
            f"{missing[:3]}"
        )
    for row in frame.itertuples(index=False):
        resolved_source = lookup[str(row.image_path)][0]
        if resolved_source != str(row.source_image):
            raise AssertionError(
                f"Source mismatch for {row.image_path}: "
                f"{resolved_source} != {row.source_image}"
            )
    return {path: lookup[path] for path in active_paths}


def build_fold_train_edge_side_counts(
    frame: pd.DataFrame,
    competition_root: Path,
) -> list[list[int]]:
    """Count XML box contacts using only the active fold-train rows.

    The four columns are left, top, right and bottom.  This is a train-only
    augmentation prior; callers must pass the already source-isolated training
    frame rather than the complete manifest.
    """
    if frame.empty or frame.fold.nunique() > 4:
        raise AssertionError("Edge profile requires an active fold-train frame")
    if frame.image_path.astype(str).str.startswith("test/").any():
        raise AssertionError("Anonymous test row entered the edge profile")
    source_lookup = build_official_source_crop_lookup(frame, competition_root)
    xml_dir = competition_root / "scripts" / "hyena_xml"
    source_sizes: dict[str, tuple[int, int]] = {}
    for source_group in sorted(frame.source_group.astype(str).unique()):
        root = ET.parse(xml_dir / f"{source_group}.xml").getroot()
        width = int(root.findtext("size/width") or 0)
        height = int(root.findtext("size/height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid XML size for {source_group}")
        source_sizes[source_group] = (width, height)

    counts = np.zeros((3, 4), dtype=np.int64)
    for row in frame.itertuples(index=False):
        part_index = int(row.part_index)
        if part_index not in (0, 1, 2):
            raise AssertionError(f"Unexpected part index: {part_index}")
        _, (xmin, ymin, xmax, ymax) = source_lookup[str(row.image_path)]
        width, height = source_sizes[str(row.source_group)]
        counts[part_index] += np.asarray(
            (
                xmin <= 1,
                ymin <= 1,
                xmax >= width - 1,
                ymax >= height - 1,
            ),
            dtype=np.int64,
        )
    if np.any(counts.sum(axis=1) <= 0):
        raise AssertionError("A crop part has no fold-train edge contacts")
    return counts.tolist()


def released_crop_geometry_features(
    frame: pd.DataFrame,
    competition_root: Path,
    *,
    allow_anonymous: bool = False,
) -> np.ndarray:
    """Read deployable pre-resize geometry from only the supplied crop rows."""
    if frame.empty:
        raise AssertionError("Released-crop geometry requires non-empty rows")
    if (
        not allow_anonymous
        and frame.image_path.astype(str).str.startswith("test/").any()
    ):
        raise AssertionError("Anonymous test row entered training geometry")
    records: list[tuple[float, float]] = []
    for image_path in frame.image_path.astype(str):
        path = competition_root / image_path
        with Image.open(path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid released crop geometry: {path}")
        records.append(
            (
                math.log(width * height),
                math.log(width / height),
            )
        )
    features = np.asarray(records, dtype=np.float32)
    if features.shape != (len(frame), len(GEOMETRY_FEATURE_NAMES)):
        raise AssertionError("Released-crop geometry shape changed")
    if not np.isfinite(features).all():
        raise FloatingPointError("Released-crop geometry is non-finite")
    return features


def build_crop_geometry_stats(
    frame: pd.DataFrame,
    competition_root: Path,
) -> dict[str, object]:
    """Fit geometry normalization on an already isolated training frame."""
    features = released_crop_geometry_features(frame, competition_root)
    mean = features.mean(axis=0, dtype=np.float64)
    std = features.std(axis=0, dtype=np.float64)
    if np.any(std <= 1e-8) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Degenerate fold-train crop geometry statistics")
    return {
        "feature_names": list(GEOMETRY_FEATURE_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "rows": int(len(frame)),
    }


class HyenaDataset(Dataset[tuple[torch.Tensor, int, int, int, int]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        competition_root: Path,
        image_size: int,
        training: bool,
        augmentation_profile: str = "default",
        geometry_stats: dict[str, object] | None = None,
        foreground_mask_root: Path | None = None,
        foreground_mode: str = "none",
        return_foreground_mask: bool = False,
        validated_foreground_index: pd.DataFrame | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.competition_root = competition_root
        self.training = bool(training)
        if foreground_mode not in FOREGROUND_MODES:
            raise ValueError(f"Unknown foreground mode: {foreground_mode}")
        if not training and foreground_mode not in {
            "none",
            "sam_view",
            "sam_part_view",
        }:
            raise ValueError(
                "Only the preregistered deterministic SAM view is allowed "
                "in evaluation data"
            )
        if (foreground_mask_root is None) != (foreground_mode == "none"):
            raise ValueError("Foreground mask root/mode must be enabled together")
        if validated_foreground_index is not None and foreground_mask_root is None:
            raise ValueError("Validated foreground index requires a mask root")
        if validated_foreground_index is not None and training:
            raise ValueError("Training cannot bypass the training-mask validator")
        self.foreground_mode = foreground_mode
        self.return_foreground_mask = bool(return_foreground_mask)
        if self.return_foreground_mask and foreground_mode == "none":
            raise ValueError("Returning a foreground mask requires a SAM view")
        self.foreground_mask_root = (
            foreground_mask_root.resolve()
            if foreground_mask_root is not None
            else None
        )
        self.foreground_records: list[dict[str, object]] | None = None
        if self.foreground_mask_root is not None:
            if validated_foreground_index is None:
                foreground_index, _ = validate_foreground_mask_artifact(
                    self.frame,
                    self.foreground_mask_root,
                    verify_mask_hashes=False,
                )
            else:
                foreground_index = validated_foreground_index.copy()
                expected = self.frame[
                    ["sample_index", "image_path", "part", "part_index"]
                ].sort_values("sample_index").reset_index(drop=True)
                actual = foreground_index[
                    ["sample_index", "image_path", "part", "part_index"]
                ].sort_values("sample_index").reset_index(drop=True)
                for column in ("image_path", "part"):
                    expected[column] = expected[column].astype(str)
                    actual[column] = actual[column].astype(str)
                if not expected.equals(actual):
                    raise AssertionError(
                        "Prevalidated evaluation foreground index/frame mismatch"
                    )
                if not actual.image_path.astype(str).str.startswith("test/").all():
                    raise AssertionError(
                        "Prevalidated evaluation foreground index is not anonymous test"
                    )
            lookup = foreground_index.set_index("sample_index").to_dict("index")
            self.foreground_records = [
                lookup[int(sample_index)]
                for sample_index in self.frame.sample_index.astype(int)
            ]
        self.source_crop_jitter = (
            training
            and augmentation_profile
            in {
                "arbase_source_jitter",
                "arbase_source_jitter_radio",
                "arbase_source_jitter_tips",
                "arbase_source_jitter_quality",
                "arbase_source_jitter_quality_body",
            }
        )
        self.source_padding_mean = (
            OPENAI_CLIP_MEAN
            if augmentation_profile == "arbase_source_jitter_radio"
            else IMAGENET_MEAN
        )
        self.quality_normalization = augmentation_profile in {
            "arbase_source_jitter_quality",
            "arbase_source_jitter_quality_body",
        }
        self.quality_body_only = (
            augmentation_profile == "arbase_source_jitter_quality_body"
        )
        base_profile = (
            "arbase"
            if augmentation_profile
            in {
                "arbase_head",
                "arbase_source_jitter",
                "arbase_source_jitter_quality",
                "arbase_source_jitter_quality_body",
            }
            else (
                "arbase_radio"
                if augmentation_profile == "arbase_source_jitter_radio"
                else (
                    "arbase_unit"
                    if augmentation_profile == "arbase_source_jitter_tips"
                    else augmentation_profile
                )
            )
        )
        self.transform = build_transform(
            image_size,
            training,
            profile=base_profile,
        )
        self.head_transform = (
            build_transform(image_size, True, profile="arbase_head")
            if training and augmentation_profile == "arbase_head"
            else None
        )
        if self.source_crop_jitter:
            source_lookup = build_official_source_crop_lookup(
                self.frame, competition_root
            )
            self.source_crops = [
                source_lookup[str(path)]
                for path in self.frame.image_path.astype(str)
            ]
        else:
            self.source_crops = None
        self.geometry_features: np.ndarray | None = None
        if geometry_stats is not None:
            if geometry_stats.get("feature_names") != list(GEOMETRY_FEATURE_NAMES):
                raise ValueError("Unexpected released-crop geometry features")
            mean = np.asarray(geometry_stats.get("mean"), dtype=np.float32)
            std = np.asarray(geometry_stats.get("std"), dtype=np.float32)
            if mean.shape != (2,) or std.shape != (2,) or np.any(std <= 0):
                raise ValueError("Invalid released-crop geometry normalization")
            raw_geometry = released_crop_geometry_features(
                self.frame,
                competition_root,
                allow_anonymous=not training,
            )
            self.geometry_features = (raw_geometry - mean) / std
            if not np.isfinite(self.geometry_features).all():
                raise FloatingPointError("Standardized crop geometry is non-finite")
        sources = self.frame.source_group.astype(str)
        source_map = {value: index for index, value in enumerate(sorted(sources.unique()))}
        self.source_codes = sources.map(source_map).astype(int).to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    def _perturbed_source_crop(self, index: int) -> np.ndarray:
        if self.source_crops is None:
            raise AssertionError("Source crop lookup is not initialized")
        source_path, (xmin, ymin, xmax, ymax) = self.source_crops[index]
        path = self.competition_root / source_path
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(path)
        box_width = xmax - xmin
        box_height = ymax - ymin
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Invalid XML box for {path}")
        scale = random.uniform(0.90, 1.15)
        center_x = 0.5 * (xmin + xmax) + random.uniform(-0.04, 0.04) * box_width
        center_y = 0.5 * (ymin + ymax) + random.uniform(-0.04, 0.04) * box_height
        half_width = 0.5 * scale * box_width
        half_height = 0.5 * scale * box_height
        left = math.floor(center_x - half_width)
        top = math.floor(center_y - half_height)
        right = math.ceil(center_x + half_width)
        bottom = math.ceil(center_y + half_height)
        height, width = source.shape[:2]
        source_left = max(0, left)
        source_top = max(0, top)
        source_right = min(width, right)
        source_bottom = min(height, bottom)
        crop = source[source_top:source_bottom, source_left:source_right]
        if crop.size == 0:
            raise ValueError(f"Empty perturbed crop for {path}")
        padding = (
            source_top - top,
            bottom - source_bottom,
            source_left - left,
            right - source_right,
        )
        if any(value > 0 for value in padding):
            crop = cv2.copyMakeBorder(
                crop,
                *padding,
                borderType=cv2.BORDER_CONSTANT,
                value=tuple(
                    round(value * 255)
                    for value in reversed(self.source_padding_mean)
                ),
            )
        return crop

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        used_source_jitter = self.source_crop_jitter and random.random() < 0.5
        if used_source_jitter:
            image = self._perturbed_source_crop(index)
        else:
            path = self.competition_root / str(row.image_path)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
        foreground_mask: np.ndarray | None = None
        foreground_valid = False
        foreground_augmented = False
        if self.foreground_records is not None:
            foreground_record = self.foreground_records[index]
            indexed_path = str(foreground_record["image_path"])
            if indexed_path != str(row.image_path):
                raise AssertionError("Foreground mask row/path alignment changed")
            foreground_valid = bool(foreground_record["valid"]) and not used_source_jitter
            if foreground_valid:
                if self.foreground_mask_root is None:
                    raise AssertionError("Foreground mask root disappeared")
                mask_path = (
                    self.foreground_mask_root
                    / "masks"
                    / f"{int(row.sample_index):06d}.png"
                )
                loaded = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if loaded is None:
                    raise FileNotFoundError(mask_path)
                if loaded.shape != image.shape[:2]:
                    raise AssertionError(
                        f"Foreground mask/image shape differs: {mask_path}"
                    )
                foreground_mask = loaded > 127
                if self.foreground_mode == "sam_bg" and random.random() < 0.5:
                    image = foreground_safe_background_blur(
                        image,
                        foreground_mask,
                        random.uniform(0.015, 0.035),
                    )
                    foreground_augmented = True
                elif self.foreground_mode == "sam_view":
                    image = foreground_safe_background_blur(
                        image,
                        foreground_mask,
                        0.035,
                    )
                    foreground_augmented = True
                elif self.foreground_mode == "sam_part_view":
                    part_index = int(row.part_index)
                    if part_index == 0:
                        if self.training and random.random() < 0.5:
                            image = foreground_safe_background_blur(
                                image,
                                foreground_mask,
                                random.uniform(0.015, 0.035),
                            )
                            foreground_augmented = True
                    elif part_index in {1, 2}:
                        image = foreground_safe_background_blur(
                            image,
                            foreground_mask,
                            0.035,
                        )
                        foreground_augmented = True
                    else:
                        raise AssertionError(
                            f"Unexpected organizer part index: {part_index}"
                        )
            else:
                foreground_mask = np.zeros(image.shape[:2], dtype=bool)
        if self.quality_normalization:
            image = apply_quality_view_bgr(
                image,
                int(row.part_index),
                body_only=self.quality_body_only,
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = (
            self.head_transform
            if self.head_transform is not None and int(row.part_index) == 0
            else self.transform
        )
        if foreground_mask is None:
            transformed = transform(image=image)
        else:
            transformed = transform(
                image=image, mask=np.uint8(foreground_mask) * 255
            )
        tensor = transformed["image"]
        record = (
            tensor,
            int(row.label_index),
            int(row.part_index),
            int(row.sample_index),
            int(self.source_codes[index]),
        )
        if self.geometry_features is not None:
            record = (
                *record,
                torch.from_numpy(self.geometry_features[index].copy()),
            )
        if foreground_mask is not None and (
            self.training or self.return_foreground_mask
        ):
            transformed_mask = transformed["mask"]
            if not isinstance(transformed_mask, torch.Tensor):
                transformed_mask = torch.from_numpy(np.asarray(transformed_mask))
            record = (*record, transformed_mask.gt(127))
            if self.training:
                record = (
                    *record,
                    torch.tensor(foreground_valid, dtype=torch.bool),
                    torch.tensor(foreground_augmented, dtype=torch.bool),
                    torch.tensor(used_source_jitter, dtype=torch.bool),
                )
        return record


class IdentityPartBatchSampler(Sampler[list[int]]):
    """Uniform identities with part- and source-diverse samples per identity."""

    def __init__(
        self,
        frame: pd.DataFrame,
        identities_per_batch: int,
        images_per_identity: int,
        seed: int,
        same_part_group: bool = False,
        balanced_cross_part_group: bool = False,
        source_paired_cross_part_group: bool = False,
        head_primary_cross_part_group: bool = False,
        body_primary_cross_part_group: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.identities_per_batch = identities_per_batch
        self.images_per_identity = images_per_identity
        self.seed = seed
        self.same_part_group = same_part_group
        self.balanced_cross_part_group = balanced_cross_part_group
        self.source_paired_cross_part_group = source_paired_cross_part_group
        self.head_primary_cross_part_group = head_primary_cross_part_group
        self.body_primary_cross_part_group = body_primary_cross_part_group
        if sum(
            (
                self.same_part_group,
                self.balanced_cross_part_group,
                self.source_paired_cross_part_group,
                self.head_primary_cross_part_group,
                self.body_primary_cross_part_group,
            )
        ) > 1:
            raise ValueError("Sampler profiles are mutually exclusive")
        self.epoch = 0
        batch_size = identities_per_batch * images_per_identity
        self.steps = math.ceil(len(self.frame) / batch_size)
        self.identities = np.asarray(sorted(self.frame.label_index.unique()))
        if len(self.identities) < identities_per_batch:
            raise ValueError("Not enough identities for one batch")
        self.records: dict[int, dict[str, Any]] = {}
        for identity, group in self.frame.groupby("label_index", sort=True):
            positions = group.index.to_numpy(dtype=np.int64)
            self.records[int(identity)] = {
                "positions": positions,
                "by_part": {
                    int(part): part_group.index.to_numpy(dtype=np.int64)
                    for part, part_group in group.groupby("part_index")
                },
                "source": self.frame.loc[positions, "source_group"].astype(str).to_dict(),
                "source_pairs": [
                    (
                        source_group.index[source_group.part_index.eq(0)].to_numpy(
                            dtype=np.int64
                        ),
                        source_group.index[source_group.part_index.isin((1, 2))].to_numpy(
                            dtype=np.int64
                        ),
                    )
                    for _, source_group in group.groupby("source_group")
                    if source_group.part_index.eq(0).any()
                    and source_group.part_index.isin((1, 2)).any()
                ],
            }

    def __len__(self) -> int:
        return self.steps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _identity_stream(self, rng: np.random.Generator) -> np.ndarray:
        needed = self.steps * self.identities_per_batch
        chunks: list[np.ndarray] = []
        while sum(len(chunk) for chunk in chunks) < needed:
            chunks.append(rng.permutation(self.identities))
        return np.concatenate(chunks)[:needed]

    def _pick(self, identity: int, rng: np.random.Generator) -> list[int]:
        record = self.records[identity]
        chosen: list[int] = []
        used_sources: set[str] = set()

        def choose(candidates: np.ndarray) -> int:
            unused = [
                int(index)
                for index in candidates
                if record["source"][int(index)] not in used_sources
            ]
            pool = unused if unused else [int(index) for index in candidates]
            selected = int(rng.choice(pool))
            used_sources.add(record["source"][selected])
            return selected

        parts = list(record["by_part"])
        rng.shuffle(parts)
        if self.same_part_group:
            # Metric positives must share the queried crop domain. Selecting
            # one part uniformly per identity occurrence preserves balanced
            # identity exposure while making all K views useful to the
            # source-aware same-part contrastive/triplet objectives.
            candidates = record["by_part"][parts[0]]
            while len(chosen) < self.images_per_identity:
                chosen.append(choose(candidates))
            return chosen
        for part in parts:
            if len(chosen) == self.images_per_identity:
                break
            chosen.append(choose(record["by_part"][part]))
        while len(chosen) < self.images_per_identity:
            # Prefer adding a same-part positive from a new source when possible.
            candidate_parts = parts.copy()
            rng.shuffle(candidate_parts)
            selected = None
            for part in candidate_parts:
                candidates = record["by_part"][part]
                if any(record["source"][int(i)] not in used_sources for i in candidates):
                    selected = choose(candidates)
                    break
            chosen.append(choose(record["positions"]) if selected is None else selected)
        return chosen

    def _pick_balanced_cross_part(
        self,
        identity: int,
        rng: np.random.Generator,
        batch_counts: np.ndarray,
        batch_target: np.ndarray,
    ) -> list[int]:
        """Keep all available parts, then assign repeats to batch deficits."""
        record = self.records[identity]
        chosen: list[int] = []
        chosen_parts: list[int] = []
        used_sources: set[str] = set()

        def choose(candidates: np.ndarray) -> int:
            unused = [
                int(index)
                for index in candidates
                if record["source"][int(index)] not in used_sources
            ]
            pool = unused if unused else [int(index) for index in candidates]
            selected = int(rng.choice(pool))
            used_sources.add(record["source"][selected])
            return selected

        parts = list(record["by_part"])
        rng.shuffle(parts)
        for part in parts:
            if len(chosen) == self.images_per_identity:
                break
            chosen.append(choose(record["by_part"][part]))
            chosen_parts.append(part)
        while len(chosen) < self.images_per_identity:
            source_diverse_parts = [
                part
                for part in parts
                if any(
                    record["source"][int(index)] not in used_sources
                    for index in record["by_part"][part]
                )
            ]
            candidates = source_diverse_parts if source_diverse_parts else parts
            local_counts = np.bincount(chosen_parts, minlength=3)
            deficits = batch_target - batch_counts - local_counts
            largest = max(int(deficits[part]) for part in candidates)
            deficit_parts = [
                part for part in candidates if int(deficits[part]) == largest
            ]
            selected_part = int(rng.choice(deficit_parts))
            chosen.append(choose(record["by_part"][selected_part]))
            chosen_parts.append(selected_part)
        return chosen

    def _pick_source_paired_cross_part(
        self,
        identity: int,
        rng: np.random.Generator,
        batch_counts: np.ndarray,
        batch_target: np.ndarray,
    ) -> list[int]:
        """Reserve a same-source head/body pair, then fill source-diverse views."""
        record = self.records[identity]
        pairs: list[tuple[np.ndarray, np.ndarray]] = record["source_pairs"]
        if not pairs:
            return self._pick_balanced_cross_part(
                identity, rng, batch_counts, batch_target
            )
        head_pool, body_pool = pairs[int(rng.integers(len(pairs)))]
        head = int(rng.choice(head_pool))
        body = int(rng.choice(body_pool))
        chosen = [head, body]
        chosen_parts = [0, int(self.frame.iloc[body].part_index)]
        used_positions = set(chosen)
        used_sources = {record["source"][head]}

        while len(chosen) < self.images_per_identity:
            local_counts = np.bincount(chosen_parts, minlength=3)
            deficits = batch_target - batch_counts - local_counts
            candidate_parts = list(record["by_part"])
            largest = max(int(deficits[part]) for part in candidate_parts)
            preferred_parts = [
                part for part in candidate_parts if int(deficits[part]) == largest
            ]
            rng.shuffle(preferred_parts)
            selected: int | None = None
            for require_new_source in (True, False):
                for part in preferred_parts + [
                    p for p in candidate_parts if p not in preferred_parts
                ]:
                    pool = [
                        int(index)
                        for index in record["by_part"][part]
                        if int(index) not in used_positions
                        and (
                            not require_new_source
                            or record["source"][int(index)] not in used_sources
                        )
                    ]
                    if pool:
                        selected = int(rng.choice(pool))
                        break
                if selected is not None:
                    break
            if selected is None:
                selected = int(rng.choice(record["positions"]))
            chosen.append(selected)
            chosen_parts.append(int(self.frame.iloc[selected].part_index))
            used_positions.add(selected)
            used_sources.add(record["source"][selected])
        return chosen

    def _pick_head_primary_cross_part(
        self,
        identity: int,
        rng: np.random.Generator,
        batch_counts: np.ndarray,
        batch_target: np.ndarray,
    ) -> list[int]:
        """Reserve two head views while retaining both body domains when available."""
        record = self.records[identity]
        if 0 not in record["by_part"]:
            return self._pick_balanced_cross_part(
                identity, rng, batch_counts, batch_target
            )
        chosen: list[int] = []
        chosen_parts: list[int] = []
        used_positions: set[int] = set()
        used_sources: set[str] = set()

        def choose(part: int) -> int:
            candidates = [
                int(index)
                for index in record["by_part"][part]
                if int(index) not in used_positions
                and record["source"][int(index)] not in used_sources
            ]
            if not candidates:
                candidates = [
                    int(index)
                    for index in record["by_part"][part]
                    if int(index) not in used_positions
                ]
            if not candidates:
                candidates = [int(index) for index in record["by_part"][part]]
            selected = int(rng.choice(candidates))
            chosen.append(selected)
            chosen_parts.append(part)
            used_positions.add(selected)
            used_sources.add(record["source"][selected])
            return selected

        # Two independently augmented head observations are the fixed primary
        # domain. Source diversity is preferred, but low-support identities are
        # still retained rather than removed from training.
        choose(0)
        choose(0)
        for body_part in (1, 2):
            if body_part in record["by_part"] and len(chosen) < self.images_per_identity:
                choose(body_part)
        while len(chosen) < self.images_per_identity:
            available_parts = list(record["by_part"])
            local_counts = np.bincount(chosen_parts, minlength=3)
            deficits = batch_target - batch_counts - local_counts
            largest = max(int(deficits[part]) for part in available_parts)
            preferred = [
                part for part in available_parts if int(deficits[part]) == largest
            ]
            choose(int(rng.choice(preferred)))
        return chosen

    def _pick_body_primary_cross_part(
        self,
        identity: int,
        rng: np.random.Generator,
        batch_counts: np.ndarray,
        batch_target: np.ndarray,
    ) -> list[int]:
        """Keep each available domain, then spend the repeat on body."""
        record = self.records[identity]
        available_parts = list(record["by_part"])
        body_parts = [part for part in (1, 2) if part in record["by_part"]]
        if not body_parts:
            return self._pick_balanced_cross_part(
                identity, rng, batch_counts, batch_target
            )
        chosen: list[int] = []
        chosen_parts: list[int] = []
        used_positions: set[int] = set()
        used_sources: set[str] = set()

        def choose(part: int) -> int:
            candidates = [
                int(index)
                for index in record["by_part"][part]
                if int(index) not in used_positions
                and record["source"][int(index)] not in used_sources
            ]
            if not candidates:
                candidates = [
                    int(index)
                    for index in record["by_part"][part]
                    if int(index) not in used_positions
                ]
            if not candidates:
                candidates = [int(index) for index in record["by_part"][part]]
            selected = int(rng.choice(candidates))
            chosen.append(selected)
            chosen_parts.append(part)
            used_positions.add(selected)
            used_sources.add(record["source"][selected])
            return selected

        # Preserve all available organizer domains before allocating the
        # repeated fourth view.  The order is fixed only at the domain level;
        # the concrete observation remains seeded and source-diverse.
        for part in (0, 1, 2):
            if part in record["by_part"] and len(chosen) < self.images_per_identity:
                choose(part)
        while len(chosen) < self.images_per_identity:
            local_counts = np.bincount(chosen_parts, minlength=3)
            candidates = body_parts if body_parts else available_parts
            deficits = batch_target - batch_counts - local_counts
            largest = max(int(deficits[part]) for part in candidates)
            preferred = [
                part for part in candidates if int(deficits[part]) == largest
            ]
            choose(int(rng.choice(preferred)))
        return chosen

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        stream = self._identity_stream(rng)
        for step in range(self.steps):
            identities = stream[
                step * self.identities_per_batch : (step + 1) * self.identities_per_batch
            ]
            batch: list[int] = []
            if (
                self.balanced_cross_part_group
                or self.source_paired_cross_part_group
                or self.head_primary_cross_part_group
                or self.body_primary_cross_part_group
            ):
                batch_size = self.identities_per_batch * self.images_per_identity
                if self.head_primary_cross_part_group:
                    batch_target = np.asarray(
                        [batch_size // 2, batch_size // 4, batch_size // 4],
                        dtype=np.int64,
                    )
                elif self.body_primary_cross_part_group:
                    head_target = batch_size // 4
                    remaining = batch_size - head_target
                    batch_target = np.asarray(
                        [head_target, remaining // 2, remaining - remaining // 2],
                        dtype=np.int64,
                    )
                else:
                    batch_target = np.full(3, batch_size // 3, dtype=np.int64)
                    for offset in range(batch_size % 3):
                        batch_target[(self.epoch + step + offset) % 3] += 1
                batch_counts = np.zeros(3, dtype=np.int64)
                for identity in identities:
                    selected = (
                        self._pick_head_primary_cross_part(
                            int(identity), rng, batch_counts, batch_target
                        )
                        if self.head_primary_cross_part_group
                        else self._pick_body_primary_cross_part(
                            int(identity), rng, batch_counts, batch_target
                        )
                        if self.body_primary_cross_part_group
                        else self._pick_source_paired_cross_part(
                            int(identity), rng, batch_counts, batch_target
                        )
                        if self.source_paired_cross_part_group
                        else self._pick_balanced_cross_part(
                            int(identity), rng, batch_counts, batch_target
                        )
                    )
                    batch.extend(selected)
                    batch_counts += np.bincount(
                        self.frame.iloc[selected].part_index.to_numpy(
                            dtype=np.int64
                        ),
                        minlength=3,
                    )
            else:
                for identity in identities:
                    batch.extend(self._pick(int(identity), rng))
            rng.shuffle(batch)
            yield batch


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_train_loader(
    frame: pd.DataFrame,
    competition_root: Path,
    image_size: int,
    workers: int,
    identities_per_batch: int,
    images_per_identity: int,
    seed: int,
    augmentation_profile: str = "default",
    sampler_profile: str = "cross_part",
    geometry_stats: dict[str, object] | None = None,
    foreground_mask_root: Path | None = None,
    foreground_mode: str = "none",
) -> tuple[DataLoader, IdentityPartBatchSampler]:
    if sampler_profile not in {
        "cross_part",
        "same_part",
        "balanced_cross_part",
        "source_paired_cross_part",
        "head_primary_cross_part",
        "body_primary_cross_part",
    }:
        raise ValueError(f"Unknown sampler profile: {sampler_profile}")
    sampler = IdentityPartBatchSampler(
        frame,
        identities_per_batch=identities_per_batch,
        images_per_identity=images_per_identity,
        seed=seed,
        same_part_group=sampler_profile == "same_part",
        balanced_cross_part_group=sampler_profile == "balanced_cross_part",
        source_paired_cross_part_group=(
            sampler_profile == "source_paired_cross_part"
        ),
        head_primary_cross_part_group=(
            sampler_profile == "head_primary_cross_part"
        ),
        body_primary_cross_part_group=(
            sampler_profile == "body_primary_cross_part"
        ),
    )
    loader = DataLoader(
        HyenaDataset(
            frame,
            competition_root,
            image_size,
            training=True,
            augmentation_profile=augmentation_profile,
            geometry_stats=geometry_stats,
            foreground_mask_root=foreground_mask_root,
            foreground_mode=foreground_mode,
        ),
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    return loader, sampler


def make_eval_loader(
    frame: pd.DataFrame,
    competition_root: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
    augmentation_profile: str = "default",
    geometry_stats: dict[str, object] | None = None,
    foreground_mask_root: Path | None = None,
    foreground_mode: str = "none",
    return_foreground_mask: bool = False,
    validated_foreground_index: pd.DataFrame | None = None,
) -> DataLoader:
    return DataLoader(
        HyenaDataset(
            frame,
            competition_root,
            image_size,
            training=False,
            augmentation_profile=augmentation_profile,
            geometry_stats=geometry_stats,
            foreground_mask_root=foreground_mask_root,
            foreground_mode=foreground_mode,
            return_foreground_mask=return_foreground_mask,
            validated_foreground_index=validated_foreground_index,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
