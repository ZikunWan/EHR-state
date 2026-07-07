#!/bin/bash
set -e

python preprocess/pds/generate_patient_splits.py \
    --root_dir "/data/zikun_workspace/input/tables/PDS" \
    --output_path "/data/zikun_workspace/input/tasks/classification/PDS/indices/patient_splits_811.json" \
    --train_ratio 0.8 \
    --val_ratio 0.1
