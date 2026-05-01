#!/bin/bash
set -e
python src/train.py --config configs/base.yaml "$@"
