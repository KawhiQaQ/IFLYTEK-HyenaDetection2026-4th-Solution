from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch


EXPECTED_MANIFEST_SHA256 = (
    "0470a4bb4630061ef06e92c50a74cb15ba457e476714294872ec426239a1ee6f"
)
EXPECTED_TEMPLATE_SHA256 = (
    "274c7db5bbc34763e1081dcd21b88abf0e087b761d13beb8b35fdc464d76f2f8"
)
EXPECTED_TEST_PART_ROWS = {"head": 798, "left_body": 414, "right_body": 429}
EXPECTED_SAM2_CHECKPOINT_SHA256 = (
    "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
)
EXPECTED_SAM2_CONFIG_SHA256 = (
    "f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d"
)
SAM2_SOURCE_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MIN_AREA = 0.10
MAX_AREA = 0.90
MIN_SCORE = 0.85


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", default="configs/sam2.1/sam2.1_hiera_t.yaml"
    )
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--scope",
        choices=("fold_train", "fold_valid", "all_train", "anonymous_test"),
        default="fold_train",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> pd.DataFrame:
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("Unexpected fixed-fold manifest SHA-256")
    if sha256_file(args.checkpoint) != EXPECTED_SAM2_CHECKPOINT_SHA256:
        raise AssertionError("Unexpected SAM2.1 checkpoint SHA-256")
    if sha256_file(args.config_file) != EXPECTED_SAM2_CONFIG_SHA256:
        raise AssertionError("Unexpected SAM2.1 config SHA-256")
    head = (
        __import__("subprocess")
        .check_output(
            ["git", "-C", str(args.sam2_repository), "rev-parse", "HEAD"],
            text=True,
        )
        .strip()
    )
    if head != SAM2_SOURCE_COMMIT:
        raise AssertionError(f"Unexpected SAM2 source commit: {head}")
    manifest = pd.read_csv(args.manifest)
    if len(manifest) != 4067 or manifest.fold.nunique() != 5:
        raise AssertionError("Unexpected training manifest geometry")
    if manifest.image_path.astype(str).str.startswith("test/").any():
        raise AssertionError("Anonymous test row entered the training manifest")
    if args.scope == "anonymous_test":
        if args.template is None:
            raise AssertionError("Anonymous-test masks require the official template")
        template = pd.read_csv(args.template)
        if sha256_file(args.template) != EXPECTED_TEMPLATE_SHA256:
            raise AssertionError("Unexpected official submission template SHA-256")
        if (
            list(template.columns) != ["image_id", "predicted_id"]
            or len(template) != 1641
            or template.image_id.nunique() != 1641
        ):
            raise AssertionError("Unexpected anonymous-test template geometry")
        from train_full import make_test_frame

        selected = make_test_frame(template)
        if (
            selected.part.value_counts().to_dict() != EXPECTED_TEST_PART_ROWS
            or not selected.image_path.astype(str).str.startswith("test/").all()
            or not selected.label_index.eq(-1).all()
        ):
            raise AssertionError("Anonymous-test frame geometry changed")
        missing = [
            value
            for value in selected.image_path.astype(str)
            if not (args.competition_root / value).is_file()
        ]
        if missing:
            raise FileNotFoundError(missing[0])
        return selected.sort_values("sample_index").reset_index(drop=True)
    if args.scope in {"fold_train", "fold_valid"}:
        train = manifest.loc[manifest.fold != args.fold].copy()
        valid = manifest.loc[manifest.fold == args.fold]
        if args.fold != 0 or len(train) != 3253 or len(valid) != 814:
            raise AssertionError("V2.63 mask generation is locked to fold 0")
        if set(train.source_group).intersection(valid.source_group):
            raise AssertionError("Source group overlap in fixed fold")
        selected = train if args.scope == "fold_train" else valid.copy()
    else:
        selected = manifest.copy()
        if len(selected) != 4067:
            raise AssertionError("All-data mask generation must cover 4,067 rows")
    if selected.sample_index.nunique() != len(selected):
        raise AssertionError("Duplicate selected sample index")
    missing = [
        value
        for value in selected.image_path.astype(str)
        if not (args.competition_root / value).is_file()
    ]
    if missing:
        raise FileNotFoundError(missing[0])
    return selected.sort_values("sample_index").reset_index(drop=True)


def select_candidate(
    centre_masks: np.ndarray,
    centre_scores: np.ndarray,
    box_masks: np.ndarray,
    box_scores: np.ndarray,
) -> tuple[np.ndarray | None, str, int, float, float]:
    candidates: list[tuple[float, str, int, float, np.ndarray]] = []
    for method, masks, scores in (
        ("center", centre_masks, centre_scores),
        ("box_center", box_masks, box_scores),
    ):
        for candidate_index, (mask, score) in enumerate(zip(masks, scores)):
            binary = np.asarray(mask) > 0
            area = float(binary.mean())
            if MIN_AREA <= area <= MAX_AREA:
                candidates.append(
                    (float(score), method, candidate_index, area, binary)
                )
    if not candidates:
        return None, "none", -1, float("nan"), float("nan")
    score, method, candidate_index, area, mask = max(
        candidates, key=lambda item: item[0]
    )
    if score < MIN_SCORE:
        return None, method, candidate_index, score, area
    return mask, method, candidate_index, score, area


def write_mask(path: Path, mask: np.ndarray) -> str:
    encoded_ok, encoded = cv2.imencode(".png", np.uint8(mask) * 255)
    if not encoded_ok:
        raise RuntimeError(f"Could not encode mask: {path}")
    payload = encoded.tobytes()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part, group in frame.groupby("part", sort=True):
        valid = group.loc[group.valid]
        result[str(part)] = {
            "rows": int(len(group)),
            "valid": int(len(valid)),
            "valid_fraction": float(len(valid) / len(group)),
            "score_quantiles": {
                str(key): float(value)
                for key, value in valid.score.quantile(
                    [0.0, 0.1, 0.5, 0.9, 1.0]
                ).items()
            },
            "area_quantiles": {
                str(key): float(value)
                for key, value in valid.area.quantile(
                    [0.0, 0.1, 0.5, 0.9, 1.0]
                ).items()
            },
            "methods": {
                str(key): int(value)
                for key, value in valid.method.value_counts().items()
            },
        }
    return result


def overlay_tile(
    image: np.ndarray,
    mask: np.ndarray | None,
    caption: str,
    width: int = 240,
    image_height: int = 180,
) -> np.ndarray:
    visual = image.copy()
    if mask is not None:
        tint = np.asarray([40, 220, 40], dtype=np.float32)
        visual[mask] = (
            0.35 * visual[mask].astype(np.float32) + 0.65 * tint
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            np.uint8(mask) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(visual, contours, -1, (255, 60, 20), 2)
    height, source_width = visual.shape[:2]
    scale = min(width / source_width, image_height / height)
    resized = cv2.resize(
        visual,
        (max(1, round(source_width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    tile = np.full((image_height + 30, width, 3), 240, dtype=np.uint8)
    top = (image_height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    tile[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    cv2.putText(
        tile,
        caption[:42],
        (3, image_height + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return tile


def make_contact_sheet(
    index: pd.DataFrame, competition_root: Path, mask_dir: Path, output: Path
) -> None:
    selected: list[pd.Series] = []
    for _, group in index.groupby("part", sort=True):
        invalid = group.loc[~group.valid].head(4)
        valid = group.loc[group.valid].sort_values(["score", "sample_index"])
        positions = sorted(
            set(
                [
                    0,
                    max(0, len(valid) // 5),
                    max(0, len(valid) // 2),
                    max(0, 4 * len(valid) // 5),
                    max(0, len(valid) - 1),
                ]
            )
        )
        selected.extend(row for _, row in invalid.iterrows())
        selected.extend(valid.iloc[position] for position in positions if len(valid))
    tiles: list[np.ndarray] = []
    for row in selected:
        bgr = cv2.imread(str(competition_root / str(row.image_path)))
        if bgr is None:
            raise FileNotFoundError(row.image_path)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = None
        if bool(row.valid):
            loaded = cv2.imread(
                str(mask_dir / f"{int(row.sample_index):06d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if loaded is None:
                raise FileNotFoundError(int(row.sample_index))
            mask = loaded > 127
        score = "invalid" if not bool(row.valid) else f"{row.score:.3f}/{row.area:.2f}"
        tiles.append(overlay_tile(image, mask, f"{row.part} {score}"))
    columns = 5
    blank = np.full_like(tiles[0], 240)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    sheet = np.vstack(
        [np.hstack(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    )
    cv2.imwrite(str(output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    args.competition_root = args.competition_root.resolve()
    args.manifest = args.manifest.resolve()
    if args.template is not None:
        args.template = args.template.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.config_file = args.config_file.resolve()
    args.sam2_repository = args.sam2_repository.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    train = validate_inputs(args)
    mask_dir = args.output_dir / "masks"
    partial_path = args.output_dir / "index.partial.jsonl"
    final_index_path = args.output_dir / "index.csv"
    if final_index_path.exists():
        raise FileExistsError(final_index_path)
    mask_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[int, dict[str, Any]] = {}
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            completed[int(record["sample_index"])] = record

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        args.config,
        str(args.checkpoint),
        device="cuda",
        mode="eval",
    )
    predictor = SAM2ImagePredictor(model)
    pending = train.loc[~train.sample_index.isin(completed)].copy()
    started = time.time()
    with partial_path.open("a", encoding="utf-8") as partial, torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        for start in range(0, len(pending), args.batch_size):
            batch = pending.iloc[start : start + args.batch_size]
            images: list[np.ndarray] = []
            points: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            boxes: list[np.ndarray] = []
            for row in batch.itertuples(index=False):
                bgr = cv2.imread(str(args.competition_root / str(row.image_path)))
                if bgr is None:
                    raise FileNotFoundError(row.image_path)
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                images.append(image)
                points.append(
                    np.asarray([[0.5 * width, 0.5 * height]], dtype=np.float32)
                )
                labels.append(np.asarray([1], dtype=np.int32))
                boxes.append(
                    np.asarray(
                        [
                            0.02 * width,
                            0.02 * height,
                            0.98 * width,
                            0.98 * height,
                        ],
                        dtype=np.float32,
                    )
                )
            predictor.set_image_batch(images)
            centre_masks, centre_scores, _ = predictor.predict_batch(
                point_coords_batch=points,
                point_labels_batch=labels,
                multimask_output=True,
            )
            box_masks, box_scores, _ = predictor.predict_batch(
                point_coords_batch=points,
                point_labels_batch=labels,
                box_batch=boxes,
                multimask_output=True,
            )
            for offset, row in enumerate(batch.itertuples(index=False)):
                mask, method, candidate, score, area = select_candidate(
                    centre_masks[offset],
                    centre_scores[offset],
                    box_masks[offset],
                    box_scores[offset],
                )
                mask_sha256 = ""
                if mask is not None:
                    if mask.shape != images[offset].shape[:2]:
                        raise AssertionError("SAM mask/image shape mismatch")
                    mask_sha256 = write_mask(
                        mask_dir / f"{int(row.sample_index):06d}.png", mask
                    )
                record = {
                    "sample_index": int(row.sample_index),
                    "image_path": str(row.image_path),
                    "source_group": str(row.source_group),
                    "part": str(row.part),
                    "part_index": int(row.part_index),
                    "valid": mask is not None,
                    "method": method,
                    "candidate_index": int(candidate),
                    "score": None if not math.isfinite(score) else float(score),
                    "area": None if not math.isfinite(area) else float(area),
                    "mask_sha256": mask_sha256,
                }
                partial.write(json.dumps(record, sort_keys=True) + "\n")
                partial.flush()
                completed[int(row.sample_index)] = record
            done = len(completed)
            if done % 96 < len(batch) or done == len(train):
                elapsed = time.time() - started
                print(
                    f"masks={done}/{len(train)} elapsed={elapsed:.1f}s "
                    f"peak_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}",
                    flush=True,
                )

    if set(completed) != set(train.sample_index.astype(int)):
        raise AssertionError("Mask index does not exactly cover fold-train rows")
    index = pd.DataFrame(completed.values()).sort_values("sample_index")
    expected_rows = {
        "fold_train": 3253,
        "fold_valid": 814,
        "all_train": 4067,
        "anonymous_test": 1641,
    }[args.scope]
    test_paths = index.image_path.astype(str).str.startswith("test/")
    paths_are_valid = (
        bool(test_paths.all())
        if args.scope == "anonymous_test"
        else not bool(test_paths.any())
    )
    if len(index) != expected_rows or not paths_are_valid:
        raise AssertionError("Invalid completed mask index")
    index.to_csv(final_index_path, index=False)
    summary = {
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "template_sha256": (
            EXPECTED_TEMPLATE_SHA256 if args.scope == "anonymous_test" else None
        ),
        "sam2_source_commit": SAM2_SOURCE_COMMIT,
        "sam2_checkpoint_sha256": EXPECTED_SAM2_CHECKPOINT_SHA256,
        "sam2_config_sha256": EXPECTED_SAM2_CONFIG_SHA256,
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
        "scope": args.scope,
        "fold": args.fold if args.scope in {"fold_train", "fold_valid"} else None,
        "rows": len(index),
        "valid": int(index.valid.sum()),
        "valid_fraction": float(index.valid.mean()),
        "minimum_score": MIN_SCORE,
        "area_range": [MIN_AREA, MAX_AREA],
        "index_sha256": sha256_file(final_index_path),
        "by_part": summarize(index),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Do not create or inspect a test-distribution contact sheet.  The exact
    # frozen mask rule is consumed mechanically by the already selected model.
    if args.scope != "anonymous_test":
        make_contact_sheet(
            index,
            args.competition_root,
            mask_dir,
            args.output_dir / "contact_sheet.jpg",
        )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
