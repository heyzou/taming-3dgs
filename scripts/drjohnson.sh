#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python train.py -s data/drjohnson -m ./eval/drjohnson_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 5 --densification_interval 500 --mode multiplier
python render.py -m ./eval/drjohnson_budget
python metrics.py -m ./eval/drjohnson_budget
