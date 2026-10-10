#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python train.py \
  -s data/room \
  -i images_2 \
  -m ./eval/room_budget \
  --quiet \
  --eval \
  --test_iterations -1 \
  --optimizer_type sparse_adam \
  --budget 2 \
  --densification_interval 500 \
  --mode multiplier \
  --jetson
python render.py -m ./eval/room_budget --jetson
python metrics.py -m ./eval/room_budget --jetson
