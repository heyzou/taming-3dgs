#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python train.py \
  -s data/bicycle \
  -i images_4 \
  -m ./eval/bicycle_budget \
  --quiet \
  --eval \
  --test_iterations -1 \
  --optimizer_type sparse_adam \
  --budget 15 \
  --densification_interval 500 \
  --mode multiplier \

python render.py -m ./eval/bicycle_budget
python metrics.py -m ./eval/bicycle_budget
