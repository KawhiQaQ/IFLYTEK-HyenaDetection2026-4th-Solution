#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python3}
public_zip=${1:?usage: $0 /path/to/dataset_public.zip /path/to/hyena-dataset.zip [competition_root]}
hyena_zip=${2:?usage: $0 /path/to/dataset_public.zip /path/to/hyena-dataset.zip [competition_root]}
competition_root=${3:-"$root/work/competition"}

if [[ -e "$competition_root" ]] && [[ -n "$(find "$competition_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing non-empty competition root: $competition_root" >&2
  exit 73
fi

mkdir -p "$competition_root"
"$python_bin" "$root/tools/restore_official_public_bundle.py" \
  --public-zip "$public_zip" \
  --competition-root "$competition_root"
"$python_bin" "$root/tools/restore_hyena_coco.py" \
  --archive "$hyena_zip" \
  --competition-root "$competition_root"

"$python_bin" "$competition_root/scripts/extract_hyena_images.py" --apply
"$python_bin" "$competition_root/scripts/crop_hyena_images_from_xml.py" --apply

PYTHONPATH="$root/benchmark/src" "$python_bin" "$root/benchmark/src/audit_folds.py" \
  --manifest "$root/benchmark/artifacts/folds.csv" \
  --competition-root "$competition_root"

echo "Official data preparation PASS: $competition_root"
