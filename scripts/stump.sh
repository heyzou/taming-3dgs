#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python train.py \
  -s data/stump \
  -i images_4 \
  -m ./eval/stump_budget \
  --quiet \
  --eval \
  --test_iterations -1 \
  --optimizer_type sparse_adam \
  --budget 15 \
  --densification_interval 500 \
  --mode multiplier \
  --benchmark_dir ./eval/stump_budget_jetson/backward_breakdown_wonder45 \
  --jetson
# python render.py -m ./eval/stump_budget --jetson
# python metrics.py -m ./eval/stump_budget --jetson
