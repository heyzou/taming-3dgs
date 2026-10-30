#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/profile_bicycle_cpu_backward_render.py \
  --source data/bicycle \
  --images images_8 \
  --camera-index 0 \
  --warmup-runs 1 \
  --profile-runs 5 \
  --output results/bicycle_cpu_backward_render_profile.csv \
  --section-output results/bicycle_cpu_backward_render_sections_direct.csv

python scripts/create_bicycle_cpu_backward_render_timing_table.py
python scripts/plot_bicycle_cpu_backward_render_direct.py
