#!/bin/bash
set -euo pipefail

data_dir="/data/zikun_workspace/input/tables/renji/raw"
split_json_dir="/data/zikun_workspace/input/metadata/splits/renji/json"
classification_index_dir="/data/zikun_workspace/input/tasks/classification/renji"
tte_index_dir="/data/zikun_workspace/input/tasks/time_to_event/renji"
embedding_cache_dir="/data/zikun_workspace/input/cache/embeddings/renji"

python preprocess/Renji/1_generate_labels.py \
  --labeled-dir "${data_dir}/follow_ups" \
  --output-file "${data_dir}/labels.csv" \
  --patient-info-file "${data_dir}/患儿基本信息总表251023_含免疫事件.csv"

python preprocess/Renji/2_generate_shots.py \
  --data_dir "${data_dir}" \
  --save_dir "${split_json_dir}"

python preprocess/Renji/generate_classification_task_index.py \
  --root-dir "${data_dir}" \
  --split-json-dir "${split_json_dir}" \
  --output-dir "${classification_index_dir}"

python preprocess/Renji/generate_tte_task_index.py \
  --root-dir "${data_dir}" \
  --split-json-dir "${split_json_dir}" \
  --output-dir "${tte_index_dir}" \
  --death-horizon-days 1825 \
  --tasks death tacrolimus

python preprocess/Renji/3_generate_text_embeddings.py \
  --stage harvest \
  --root-dir "${data_dir}" \
  --cache-dir "${embedding_cache_dir}" \
  --harvest-checkpoint "${embedding_cache_dir}/unique_texts_harvested.pkl"
