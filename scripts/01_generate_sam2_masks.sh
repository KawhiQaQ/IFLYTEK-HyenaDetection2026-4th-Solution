#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python3}
competition_root=${COMPETITION_ROOT:-"$root/work/competition"}
weights_root=${WEIGHTS_ROOT:-"$root/../weights"}
sam2_repo=${SAM2_REPO:?Set SAM2_REPO to facebookresearch/sam2 at commit 2b90b9f5ceec907a1c18123530e92e794ad901a4}
sam2_checkpoint=${SAM2_CHECKPOINT:-"$weights_root/sam2/sam2.1_hiera_tiny.pt"}
sam2_config=${SAM2_CONFIG_FILE:-"$sam2_repo/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"}
mask_root=${MASK_ROOT:-"$root/work/masks"}

export PYTHONPATH="$root/src/hyenaid:$sam2_repo${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" "$root/tools/generate_sam2_masks.py" \
  --competition-root "$competition_root" \
  --manifest "$root/benchmark/artifacts/folds.csv" \
  --checkpoint "$sam2_checkpoint" \
  --config-file "$sam2_config" \
  --sam2-repository "$sam2_repo" \
  --output-dir "$mask_root/all_train" \
  --scope all_train \
  --batch-size 8

"$python_bin" "$root/tools/generate_sam2_masks.py" \
  --competition-root "$competition_root" \
  --manifest "$root/benchmark/artifacts/folds.csv" \
  --template "$competition_root/submission_template.csv" \
  --checkpoint "$sam2_checkpoint" \
  --config-file "$sam2_config" \
  --sam2-repository "$sam2_repo" \
  --output-dir "$mask_root/anonymous_test" \
  --scope anonymous_test \
  --batch-size 8

train_sha=$(sha256sum "$mask_root/all_train/index.csv" | awk '{print $1}')
test_sha=$(sha256sum "$mask_root/anonymous_test/index.csv" | awk '{print $1}')
[[ "$train_sha" == "53fbb173a1ace9a5cdbf8054f7c7b5747e9e30d639940ad13d4f2e5304be8687" ]]
[[ "$test_sha" == "7aba1a44ec180df5b6b6257807ace7a81b693e3212efdef1d7707b7f005a2750" ]]
echo "SAM2 mask preparation PASS: $mask_root"
