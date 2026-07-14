#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

mode="${1:-train}"
use_eval_dataset="${2:-true}"
data_dir="/data/zikun_workspace/mimic-iv-3.1_tabular"
index_dir="/data/zikun_workspace/input/tasks/classification/mimic_iv/indices"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/mimic_iv/text_embeddings.pt"
checkpoint_root="/data/zikun_workspace/checkpoints/classification/ehr_bench"
query_cache_dir="/data/zikun_workspace/input/cache/query_embeddings/query_classifier/mimic_iv"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
pretrained_path="/data/zikun_workspace/checkpoints/pretraining/1B"
tasks=(ED_Hospitalization ED_Inpatient_Mortality ED_ICU_Tranfer_12hour ED_Reattendance_3day ED_Critical_Outcomes Readmission_30day Readmission_60day Inpatient_Mortality LengthOfStay_3day LengthOfStay_7day ICU_Mortality_1day ICU_Mortality_2day ICU_Mortality_3day ICU_Mortality_7day ICU_Mortality_14day ICU_Stay_7day ICU_Stay_14day ICU_Readmission)

for task in "${tasks[@]}"; do
  if [[ "${mode}" == "train" ]]; then
    deepspeed --include localhost:0,1,2,3,4,5,6,7 train/classification/train_encoder_classifier.py \
      --deepspeed "ds_config_zero2.json" \
      --dataset_name ehr_bench \
      --data_dir "${data_dir}" \
      --train_info_path "${index_dir}/train/${task}.csv" \
      --val_info_path "${index_dir}/val/${task}.csv" \
      --max_train_samples 3000 \
      --max_eval_samples 1000 \
      --use_eval_dataset "${use_eval_dataset}" \
      --task_name "${task}" \
      --embedding_cache "${embedding_cache}" \
      --output_dir "${checkpoint_root}/${task}/full_tune" \
      --run_name "mimic_iv_${task}" \
      --type_vocab_file "data/type_vocab.json" \
      --query_embedding_cache "${query_cache_dir}/${task}.pt" \
      --knowledge_encoder_path "${knowledge_encoder}" \
      --knowledge_encoder_base_model_path "${base_model}" \
      --pretrained_path "${pretrained_path}" \
      --query_max_length 128 \
      --max_table_len 4096 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 32 \
      --eval_strategy steps \
      --eval_steps 100 \
      --save_strategy steps \
      --save_steps 100 \
      --save_total_limit 1 \
      --load_best_model_at_end true \
      --num_train_epochs 20 \
      --learning_rate 3e-5 \
      --bf16 true \
      --dataloader_num_workers 16 \
      --report_to wandb
  elif [[ "${mode}" == "eval" ]]; then
    CUDA_VISIBLE_DEVICES=0 python test/classification/test_encoder_classifier.py \
      --dataset_name ehr_bench \
      --data_dir "${data_dir}" \
      --sample_info_test_path "${index_dir}/test/${task}.csv" \
      --checkpoint_dir "${checkpoint_root}/${task}/full_tune" \
      --task_name "${task}" \
      --embedding_cache "${embedding_cache}" \
      --type_vocab_file "data/type_vocab.json" \
      --query_embedding_cache "${query_cache_dir}/${task}.pt" \
      --knowledge_encoder_path "${knowledge_encoder}" \
      --knowledge_encoder_base_model_path "${base_model}" \
      --query_max_length 128 \
      --max_table_len 16384 \
      --batch_size 64
  else
    echo "Usage: $0 [train|eval] [true|false]" >&2
    exit 2
  fi
done
