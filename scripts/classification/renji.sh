#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

mode="${1:-train}"
task_arg="${2:-all}"
data_dir="/data/zikun_workspace/input/tables/renji/raw"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/renji/text_embeddings.pt"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
tasks=(ALB ALP CR Glucose HB INR N_Percent PLT PT TP Uric_Acid WBC)

if [[ "${task_arg}" != "all" ]]; then
  tasks=("${task_arg}")
fi

run_train() {
  local task="$1"
  local checkpoint_dir="/data/zikun_workspace/checkpoints/classification/renji/${task}"
  local query_cache="/data/zikun_workspace/input/cache/query_embeddings/query_classifier/renji/${task}.pt"
  deepspeed --include localhost:4,5,6,7 train/classification/train_encoder_classifier.py \
    --deepspeed "ds_config_zero2.json" \
    --dataset_name renji \
    --data_dir "${data_dir}" \
    --task_name "${task}" \
    --embedding_cache "${embedding_cache}" \
    --output_dir "${checkpoint_dir}" \
    --run_name "renji_${task}" \
    --type_vocab_file "data/type_vocab.json" \
    --query_embedding_cache "${query_cache}" \
    --knowledge_encoder_path "${knowledge_encoder}" \
    --knowledge_encoder_base_model_path "${base_model}" \
    --query_max_length 128 \
      --max_table_len 16384 \
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
}

run_eval() {
  local task="$1"
  local checkpoint_dir="/data/zikun_workspace/checkpoints/classification/renji/${task}"
  local query_cache="/data/zikun_workspace/input/cache/query_embeddings/query_classifier/renji/${task}.pt"
  CUDA_VISIBLE_DEVICES=0 python test/classification/test_encoder_classifier.py \
    --dataset_name renji \
    --data_dir "${data_dir}" \
    --split test \
    --checkpoint_dir "${checkpoint_dir}" \
    --task_name "${task}" \
    --embedding_cache "${embedding_cache}" \
    --type_vocab_file "data/type_vocab.json" \
    --query_embedding_cache "${query_cache}" \
    --knowledge_encoder_path "${knowledge_encoder}" \
    --knowledge_encoder_base_model_path "${base_model}" \
    --query_max_length 128 \
    --max_table_len 16384 \
    --batch_size 64
}

if [[ "${mode}" == "train" ]]; then
  for task in "${tasks[@]}"; do
    run_train "${task}"
  done
elif [[ "${mode}" == "eval" ]]; then
  for task in "${tasks[@]}"; do
    run_eval "${task}"
  done
else
  echo "Usage: $0 [train|eval] [all|ALB|ALP|CR|Glucose|HB|INR|N_Percent|PLT|PT|TP|Uric_Acid|WBC]" >&2
  exit 2
fi
