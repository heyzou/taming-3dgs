#!/usr/bin/env bash
set -e
set -o pipefail
cd "$(dirname "$0")/.."

RESULT_FILE="./results/training_progress_wonder44.txt"
mkdir -p results
: > "$RESULT_FILE"

# Jetson benchmark on wonder44
run_scene() {
  local scene="$1"
  local images="$2"
  local budget="$3"
  local tmp_log
  local progress_line
  tmp_log=$(mktemp)
  python train.py \
    -s "data/${scene}" \
    -i "${images}" \
    -m "./eval/${scene}_budget" \
    --quiet \
    --eval \
    --test_iterations -1 \
    --optimizer_type sparse_adam \
    --budget "${budget}" \
    --densification_interval 500 \
    --mode multiplier \
    --benchmark_dir "./eval/${scene}_budget_jetson" \
    --jetson \
    2>&1 | tee "$tmp_log"

  progress_line=$(tr '\r' '\n' < "$tmp_log" | grep "Training progress" | tail -n 1 || true)
  if [[ -n "$progress_line" ]]; then
    printf '[%s] %s\n' "$scene" "$progress_line" >> "$RESULT_FILE"
  fi
  rm -f "$tmp_log"
}

run_scene bicycle images_4 15
run_scene counter images_2 2
run_scene garden images_4 15
run_scene room images_2 2
run_scene treehill images_4 15
