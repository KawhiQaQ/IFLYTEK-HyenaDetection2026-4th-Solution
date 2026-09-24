from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_folds import EXPECTED_COUNTS, audit_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/folds.csv"))
    parser.add_argument("--competition-root", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    report = audit_manifest(frame, n_splits=5)
    if report["part_counts"] != EXPECTED_COUNTS:
        raise AssertionError(
            f"Part counts differ from official values: {report['part_counts']}"
        )
    if args.competition_root is not None:
        missing = [
            path
            for path in frame["image_path"]
            if not (args.competition_root / path).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} crops; first={missing[0]}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

