#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p results

nsys profile \
  --trace=none \
  --sample=process-tree \
  --backtrace=dwarf \
  --sampling-period=137600 \
  --force-overwrite=true \
  --output=results/bicycle_cpu_backward_render_nsys \
  python scripts/profile_bicycle_cpu_backward_render.py \
    --source data/bicycle \
    --images images_8 \
    --camera-index 0 \
    --warmup-runs 1 \
    --profile-runs 0 \
    --sampling-runs 20

python scripts/analyze_bicycle_cpu_backward_render_nsys.py
python scripts/plot_bicycle_cpu_backward_render_profile.py
