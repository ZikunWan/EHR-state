#!/bin/bash
set -euo pipefail

ROOT_DIR="/data/EHR_data_public/EHRSHOT_ASSETS"
TABLE_DIR="/data/zikun_workspace/input/tables/ehrshot"
UTILS_DIR="/data/zikun_workspace/input/cache/ehrshot/utils"
CLASSIFICATION_INDEX_DIR="/data/zikun_workspace/input/tasks/classification/ehrshot"
SAMPLE_INFO_DIR="/data/zikun_workspace/input/cache/ehrshot/classification_sample_info"
PRETRAINING_INDEX_DIR="/data/zikun_workspace/input/tasks/pretraining/ehrshot/indices"

python preprocess/ehrshot/1_generate_patient_ehr.py \
  --ehrshot_csv "${ROOT_DIR}/data/ehrshot.csv" \
  --clmbr_dir "${ROOT_DIR}/models/clmbr" \
  --output_dir "${TABLE_DIR}/patients" \
  --output_utils_dir "${UTILS_DIR}"

python preprocess/ehrshot/2_generate_sample_info.py \
  --splits_path "${ROOT_DIR}/splits/person_id_map.csv" \
  --benchmark_dir "${ROOT_DIR}/benchmark" \
  --patient_dir "${TABLE_DIR}/patients" \
  --output_dir "${SAMPLE_INFO_DIR}" \
  --classification_index_dir "${CLASSIFICATION_INDEX_DIR}" \
  --pretraining_index_dir "${PRETRAINING_INDEX_DIR}"
