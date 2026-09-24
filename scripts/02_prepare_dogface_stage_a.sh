#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python3}
dogface_zip=${1:?usage: $0 /path/to/DogFaceNet_224resized.zip}
competition_root=${COMPETITION_ROOT:-"$root/work/competition"}
prepared_root=${DOGFACE_PREPARED_ROOT:-"$root/work/dogfacenet/prepared"}
output_root=${DOGFACE_STAGE_A_ROOT:-"$root/work/dogfacenet/stage_a"}

"$python_bin" "$root/tools/prepare_dogface_data.py" \
  --archive "$dogface_zip" \
  --output-root "$prepared_root" \
  --competition-root "$competition_root"

export PYTHONPATH="$root/src/hyenaid${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" "$root/tools/pretrain_dogface_stage_a.py" \
  --data-root "$prepared_root" \
  --manifest "$prepared_root/manifest.csv" \
  --dataset-audit "$prepared_root/audit.json" \
  --output-dir "$output_root" \
  --workers "${WORKERS:-8}"

echo "DogFace Stage-A preparation PASS: $output_root"
echo "checkpoint SHA-256: $(sha256sum "$output_root/external_reid_final.pt" | awk '{print $1}')"
