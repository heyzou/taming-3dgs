#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

cpu_model_dir="./eval/bicycle_budget_sequential_cpu_backward_render"
cpu_benchmark_dir="./eval/bicycle_budget_sequential_cpu_backward_render_jetson/backward_breakdown_wonder44"

python train.py \
  -s data/bicycle \
  -i images_4 \
  -m "${cpu_model_dir}" \
  --quiet \
  --eval \
  --test_iterations -1 \
  --optimizer_type sparse_adam \
  --budget 15 \
  --densification_interval 500 \
  --mode multiplier \
  --benchmark_dir "${cpu_benchmark_dir}" \
  --cpu_backward_render \
  --jetson

python eval/plot_backward_breakdown.py "${cpu_benchmark_dir}"

# The --jetson option adds the _jetson suffix to cpu_model_dir.
# python render.py -m "${cpu_model_dir}" --jetson
# python metrics.py -m "${cpu_model_dir}" --jetson
