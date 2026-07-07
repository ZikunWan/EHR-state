#!/bin/bash
set -euo pipefail

ROOT_DIR="/data/zikun_workspace/mimic-iv-3.1_tabular"
INDEX_DIR="/data/zikun_workspace/input/tasks/classification/mimic_iv/index"
TTE_INDEX_DIR="/data/zikun_workspace/input/tasks/time_to_event/mimic_iv/indices"
TASKS="ED_Hospitalization,ED_Inpatient_Mortality,ED_ICU_Tranfer_12hour,ED_Reattendance_3day,ED_Critical_Outcomes,Readmission_30day,Readmission_60day,Inpatient_Mortality,LengthOfStay_3day,LengthOfStay_7day,ICU_Mortality_1day,ICU_Mortality_2day,ICU_Mortality_3day,ICU_Mortality_7day,ICU_Mortality_14day,ICU_Stay_7day,ICU_Stay_14day,ICU_Readmission,next_token_prediction"

python preprocess/mimic_iv/4_task_sample_info_gen.py \
  --root_dir "${ROOT_DIR}" \
  --ehr_dir "${ROOT_DIR}/patients_ehr" \
  --output_path "${INDEX_DIR}/all" \
  --task "${TASKS}"

python preprocess/mimic_iv/5_generate_cohorts.py \
  --ehr_dir "${ROOT_DIR}/patients_ehr" \
  --task_index_all_dir "${INDEX_DIR}/all" \
  --patient_output_dir "${ROOT_DIR}/patient_data" \
  --task_index_output_dir "${INDEX_DIR}" \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --random_seed 42

python preprocess/mimic_iv/generate_tte_task_index.py \
  --ehr_dir "${ROOT_DIR}/patients_ehr" \
  --train_index_dir "${INDEX_DIR}/train" \
  --val_index_dir "${INDEX_DIR}/val" \
  --test_index_dir "${INDEX_DIR}/test" \
  --output_dir "${TTE_INDEX_DIR}" \
  --num_workers 32 \
  --worker_chunksize 8
