#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python3}
competition_root=${COMPETITION_ROOT:-"$root/work/competition"}
mask_root=${MASK_ROOT:-"$root/work/masks"}
output_root=${OUTPUT_ROOT:-"$root/work/full_run"}
weights_root=${WEIGHTS_ROOT:-"$root/../weights"}
hf_home=${HF_HOME:-"$weights_root/hf_home"}
dogface_checkpoint=${DOGFACE_STAGE_A_CHECKPOINT:-"$weights_root/dogface_stage_a/external_reid_final.pt"}
dogface_audit=${DOGFACE_STAGE_A_AUDIT:-"$weights_root/dogface_stage_a/checkpoint_audit.json"}
bioclip_checkpoint=${BIOCLIP_CHECKPOINT:-"$weights_root/bioclip/open_clip_pytorch_model.bin"}
workers=${WORKERS:-8}

mkdir -p "$output_root/anchor_components" "$output_root/complementary_components"
export PYTHONPATH="$root/src/hyenaid:$root/tools${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" "$root/tools/prepare_weights.py" \
  --weights-root "$weights_root" \
  --hf-home "$hf_home"
export HF_HOME="$hf_home"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

dogface_sha=$(sha256sum "$dogface_checkpoint" | awk '{print $1}')

for component in D H A B Q C; do
  destination="$output_root/anchor_components/$component"
  if [[ -e "$destination" ]] && [[ -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing non-empty component output: $destination" >&2
    exit 74
  fi
  extra=()
  if [[ "$component" == D ]]; then
    extra=(
      --external-reid-checkpoint "$dogface_checkpoint"
      --external-reid-audit "$dogface_audit"
      --expected-external-reid-sha256 "$dogface_sha"
    )
  fi
  "$python_bin" "$root/tools/train_anchor_component.py" \
    --component "$component" \
    --competition-root "$competition_root" \
    --manifest "$root/benchmark/artifacts/folds.csv" \
    --template "$competition_root/submission_template.csv" \
    --train-foreground-mask-root "$mask_root/all_train" \
    --test-foreground-mask-root "$mask_root/anonymous_test" \
    --output-dir "$destination" \
    --workers "$workers" \
    --eval-batch-size 20 \
    "${extra[@]}" \
    2>&1 | tee "$output_root/anchor_${component}.log"
done

"$python_bin" "$root/tools/fuse_anchor.py" \
  --component-root "$output_root/anchor_components" \
  --template "$competition_root/submission_template.csv" \
  --output-dir "$output_root/anchor"

"$python_bin" "$root/tools/audit_anchor.py" \
  --manifest "$root/benchmark/artifacts/folds.csv" \
  --component-root "$output_root/anchor_components" \
  --template "$competition_root/submission_template.csv" \
  --output-dir "$output_root/anchor"

for component in F K; do
  destination="$output_root/complementary_components/$component"
  if [[ -e "$destination" ]] && [[ -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing non-empty component output: $destination" >&2
    exit 75
  fi
  extra=()
  if [[ "$component" == F ]]; then
    extra=(--bioclip-checkpoint "$bioclip_checkpoint")
  fi
  "$python_bin" "$root/tools/train_complementary_component.py" \
    --component "$component" \
    --competition-root "$competition_root" \
    --manifest "$root/benchmark/artifacts/folds.csv" \
    --template "$competition_root/submission_template.csv" \
    --train-foreground-mask-root "$mask_root/all_train" \
    --test-foreground-mask-root "$mask_root/anonymous_test" \
    --output-dir "$destination" \
    --workers "$workers" \
    "${extra[@]}" \
    2>&1 | tee "$output_root/complementary_${component}.log"
done

"$python_bin" "$root/tools/build_final_submission.py" \
  --anchor-score "$output_root/anchor/fused_scores.pt" \
  --f-score "$output_root/complementary_components/F/component_scores.pt" \
  --k-score "$output_root/complementary_components/K/component_scores.pt" \
  --template "$competition_root/submission_template.csv" \
  --test-predictions "$output_root/anchor/test_predictions.csv" \
  --output-dir "$output_root/final_submission"

echo "Full training and inference PASS: $output_root/final_submission/submission.zip"
