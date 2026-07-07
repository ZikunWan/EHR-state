#!/bin/bash
set -euo pipefail

MIMIC_HOSP_DIR="/data/zikun_workspace/mimic-iv-3.1/hosp"
CDM_ROOT_DIR="/data/EHR_data_public/mimic-iv-cdm"
INDEX_DIR="/data/zikun_workspace/input/tasks/classification/mimic_iv_cdm/index"
PATIENT_DIR="/data/zikun_workspace/input/metadata/splits/mimic_iv_cdm"

python preprocess/mimic_iv_cdm/1_generate_sample_info.py \
  --root_dir "${CDM_ROOT_DIR}" \
  --output_index_dir "${INDEX_DIR}" \
  --output_patient_dir "${PATIENT_DIR}" \
  --categories "appendicitis,cholecystitis,diverticulitis,pancreatitis" \
  --train_ratio 0.8 \
  --random_seed 42

python preprocess/mimic_iv_cdm/2_generate_microbiology_mapping.py \
  --micro_events_path "${MIMIC_HOSP_DIR}/microbiologyevents.csv.gz" \
  --output_path "${CDM_ROOT_DIR}/microbiology_test_mapping.pkl"

python preprocess/mimic_iv_cdm/3_generate_icd_mapping.py \
  --mimic_hosp_path "${MIMIC_HOSP_DIR}" \
  --output_dir "${CDM_ROOT_DIR}"
