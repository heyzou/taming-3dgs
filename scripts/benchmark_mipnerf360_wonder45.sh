#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Jetson benchmark on wonder45
run_scene() {
  local scene="$1"
  local images="$2"
  local budget="$3"
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
    --jetson
}

run_scene bonsai images_2 2
run_scene flowers images_4 15
run_scene kitchen images_2 2
run_scene stump images_4 15
