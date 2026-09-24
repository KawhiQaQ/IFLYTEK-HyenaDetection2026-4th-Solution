#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python3}
output_dir=${1:-"$root/work/frozen_rebuild"}

if [[ -e "$output_dir" ]]; then
  echo "Refusing existing output directory: $output_dir" >&2
  exit 73
fi

"$python_bin" "$root/tools/build_final_submission.py" \
  --anchor-score "$root/artifacts/frozen_scores/anchor/fused_scores.pt" \
  --f-score "$root/artifacts/frozen_scores/complementary/F/component_scores.pt" \
  --k-score "$root/artifacts/frozen_scores/complementary/K/component_scores.pt" \
  --template "$root/artifacts/frozen_scores/anchor/submission.csv" \
  --test-predictions "$root/artifacts/frozen_scores/anchor/test_predictions.csv" \
  --output-dir "$output_dir" \
  --require-reference-hash

echo "Frozen-score exact reproduction PASS: $output_dir/submission.csv"
