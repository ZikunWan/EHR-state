#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

mode="${1:-train}"
use_eval_dataset="${2:-true}"
data_dir="/data/zikun_workspace/input/tables/PDS"
patient_split_path="/data/zikun_workspace/input/tasks/classification/PDS"
trial_id="102,103,105,118,119,121,122,127,128,149"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/PDS/text_embeddings.pt"
checkpoint_root="/data/zikun_workspace/checkpoints/classification/pds"
query_cache_dir="/data/zikun_workspace/input/cache/query_embeddings/query_classifier/PDS"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
pretrained_path="/data/zikun_workspace/checkpoints/pretraining/1B"
tasks=(severe_outcome adverse_event_next_visit)

for task in "${tasks[@]}"; do
  task_trial_id="${trial_id}"
  if [[ "${task}" == "adverse_event_next_visit" ]]; then
    task_trial_id="102,103,105,118,119,121,122,127,128"
  fi
  if [[ "${mode}" == "train" ]]; then
    deepspeed --include localhost:0,1,2,3,4,5,6,7 train/classification/train_encoder_classifier.py \
      --deepspeed "ds_config_zero2.json" \
      --dataset_name pds \
      --data_dir "${data_dir}" \
      --patient_split_path "${patient_split_path}" \
      --trial_id "${task_trial_id}" \
      --task_name "${task}" \
      --use_eval_dataset "${use_eval_dataset}" \
      --embedding_cache "${embedding_cache}" \
      --output_dir "${checkpoint_root}/${task}/full_tune" \
      --run_name "pds_${task}" \
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
      --num_train_epochs 50 \
      --learning_rate 3e-5 \
      --bf16 true \
      --dataloader_num_workers 16 \
      --report_to wandb
  elif [[ "${mode}" == "eval" ]]; then
    CUDA_VISIBLE_DEVICES=0 python test/classification/test_encoder_classifier.py \
      --dataset_name pds \
      --data_dir "${data_dir}" \
      --patient_split_path "${patient_split_path}" \
      --trial_id "${task_trial_id}" \
      --split test \
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
