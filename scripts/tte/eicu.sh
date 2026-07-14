#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

mode="${1:-train}"
task_arg="${2:-all}"
data_dir="/data/EHR_data_public/eicu-crd/2.0"
processed_dir="/data/zikun_workspace/eicu-crd/processed"
index_dir="/data/zikun_workspace/input/tasks/time_to_event/eicu"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/eicu/text_embeddings.pt"
checkpoint_root="/data/zikun_workspace/checkpoints/tte/eicu"
query_cache_dir="/data/zikun_workspace/input/cache/query_embeddings/tte/eicu"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
pretrained_path="/data/zikun_workspace/checkpoints/pretraining/1B"
tasks=(Time_to_Mortality Time_to_ICU_Discharge)

if [[ "${task_arg}" != "all" ]]; then
  tasks=("${task_arg}")
fi

for task in "${tasks[@]}"; do
  if [[ "${mode}" == "train" ]]; then
    deepspeed --include localhost:0,1,2,3,4,5,6,7 train/tte/train_eicu_tte.py \
      --deepspeed "ds_config_zero2.json" \
      --data_dir "${data_dir}" \
      --processed_dir "${processed_dir}" \
      --train_info_path "${index_dir}/train/${task}.csv" \
      --val_info_path "${index_dir}/val/${task}.csv" \
      --max_train_samples 10000 \
      --task_name "${task}" \
      --embedding_cache "${embedding_cache}" \
      --output_dir "${checkpoint_root}/${task}/full_tune" \
      --run_name "eicu_${task}" \
      --type_vocab_file "data/type_vocab.json" \
      --query_embedding_cache "${query_cache_dir}/${task}.pt" \
      --knowledge_encoder_path "${knowledge_encoder}" \
      --knowledge_encoder_base_model_path "${base_model}" \
      --query_max_length 128 \
      --max_table_len 4096 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 32 \
      --eval_strategy steps \
      --eval_steps 100 \
      --save_strategy steps \
      --save_steps 100 \
      --save_total_limit 1 \
      --early_stopping_patience 10 \
      --load_best_model_at_end true \
      --num_train_epochs 100 \
      --learning_rate 3e-5 \
      --bf16 true \
      --dataloader_num_workers 16 \
      --report_to wandb \
      --pretrained_path "${pretrained_path}"
  elif [[ "${mode}" == "eval" ]]; then
    CUDA_VISIBLE_DEVICES=0 python test/tte/test_eicu_tte.py \
      --data_dir "${data_dir}" \
      --processed_dir "${processed_dir}" \
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
    echo "Usage: $0 [train|eval] [all|Time_to_Mortality|Time_to_ICU_Discharge]" >&2
    exit 2
  fi
done
