#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python train.py -s data/garden -i images_4 -m ./eval/garden_budget --quiet --eval --test_iterations -1 --optimizer_type sparse_adam --budget 15 --densification_interval 500 --mode multiplier --jetson
python render.py -m ./eval/garden_budget --jetson
python metrics.py -m ./eval/garden_budget --jetson
