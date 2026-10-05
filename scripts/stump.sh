#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python train.py -s data/stump -i images_4 -m ./eval/stump_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 15 --densification_interval 500 --mode multiplier
python render.py -m ./eval/stump_budget
python metrics.py -m ./eval/stump_budget
