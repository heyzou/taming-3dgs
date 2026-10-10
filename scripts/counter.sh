#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python train.py \
  -s data/counter \
  -i images_2 \
  -m ./eval/counter_budget \
  --quiet \
  --eval \
  --test_iterations -1 \
  --optimizer_type sparse_adam \
  --budget 2 \
  --densification_interval 500 \
  --mode multiplier \
  --benchmark_dir ./eval/bicycle_budget
# python render.py -m ./eval/counter_budget --jetson
# python metrics.py -m ./eval/counter_budget --jetson
