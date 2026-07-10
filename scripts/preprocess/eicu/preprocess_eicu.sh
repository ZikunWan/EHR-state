#!/bin/bash
set -euo pipefail

CONFIG_PATH="preprocess/eicu/config.yaml"
PROCESSED_DIR="/data/zikun_workspace/eicu-crd/processed"
CLASSIFICATION_INDEX_DIR="/data/zikun_workspace/input/tasks/classification/eicu"
TTE_INDEX_DIR="/data/zikun_workspace/input/tasks/time_to_event/eicu/indices"

python preprocess/eicu/1_build_cohorts.py \
  --config "${CONFIG_PATH}"

python preprocess/eicu/2_prepare_tasks.py \
  --config "${CONFIG_PATH}"

python preprocess/eicu/3_generate_sample_info.py \
  --config "${CONFIG_PATH}"

python preprocess/eicu/4_partition_patients.py \
  --config "${CONFIG_PATH}"

python preprocess/split_classification_task_index.py \
  --dataset eicu \
  --input-dir "${PROCESSED_DIR}" \
  --output-dir "${CLASSIFICATION_INDEX_DIR}"

python preprocess/eicu/generate_tte_task_index.py \
  --cohorts_path "${PROCESSED_DIR}/cohorts.csv" \
  --train_sample_info_path "${PROCESSED_DIR}/sample_info_train.json" \
  --val_sample_info_path "${PROCESSED_DIR}/sample_info_val.json" \
  --test_sample_info_path "${PROCESSED_DIR}/sample_info_test.json" \
  --output_dir "${TTE_INDEX_DIR}"
