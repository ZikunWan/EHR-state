#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

mode="${1:-train}"
task_arg="${2:-all}"
data_dir="/data/zikun_workspace/input/tables/renji/raw"
index_dir="/data/zikun_workspace/input/tasks/time_to_event/renji"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/renji/text_embeddings.pt"
checkpoint_root="/data/zikun_workspace/checkpoints/tte/renji"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
pretrained_path="/data/zikun_workspace/checkpoints/pretraining/1B"
tasks=(tacrolimus_abnormal death)

if [[ "${task_arg}" != "all" ]]; then
  tasks=("${task_arg}")
fi

for task in "${tasks[@]}"; do
  if [[ "${task}" == "death" ]]; then
    task_slug="death_survival"
    query_cache="/data/zikun_workspace/input/cache/query_embeddings/query_candidate/renji_death_survival_task_query_knowledge_embeddings.pt"
    train_batch_size=16
    eval_batch_size=32
  else
    task_slug="tacrolimus_abnormal_survival"
    query_cache="/data/zikun_workspace/input/cache/query_embeddings/query_candidate/renji_survival_task_query_knowledge_embeddings.pt"
    train_batch_size=32
    eval_batch_size=128
  fi

  if [[ "${mode}" == "train" ]]; then
    deepspeed --include localhost:0,1,2,3,4,5,6,7 train/tte/train_renji_tte.py \
      --deepspeed "ds_config_zero2.json" \
      --survival_task "${task}" \
      --data_dir "${data_dir}" \
      --embedding_cache "${embedding_cache}" \
      --output_dir "${checkpoint_root}/${task_slug}/full_tune" \
      --run_name "renji_${task_slug}" \
      --patient_subset_path "data/patients.json" \
      --tte_index_dir "${index_dir}" \
      --type_vocab_file "data/type_vocab.json" \
      --query_embedding_cache "${query_cache}" \
      --knowledge_encoder_path "${knowledge_encoder}" \
      --knowledge_encoder_base_model_path "${base_model}" \
      --query_max_length 128 \
      --max_table_len 4096 \
      --per_device_train_batch_size "${train_batch_size}" \
      --per_device_eval_batch_size "${eval_batch_size}" \
      --monitor_fraction 0.1 \
      --monitor_seed 42 \
      --eval_strategy steps \
      --eval_steps 100 \
      --save_strategy steps \
      --save_steps 100 \
      --save_total_limit 1 \
      --early_stopping_patience 10 \
      --load_best_model_at_end true \
      --num_train_epochs 100 \
      --learning_rate 3e-5 \
      --lr_scheduler_type cosine_with_min_lr \
      --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
      --warmup_steps 100 \
      --bf16 true \
      --dataloader_num_workers 32 \
      --report_to wandb \
      --pretrained_path "${pretrained_path}"
  elif [[ "${mode}" == "eval" ]]; then
    CUDA_VISIBLE_DEVICES=0 python test/tte/test_renji_tte.py \
      --survival_task "${task}" \
      --data_dir "${data_dir}" \
      --embedding_cache "${embedding_cache}" \
      --checkpoint_dir "${checkpoint_root}/${task_slug}/full_tune" \
      --patient_subset_path "data/patients.json" \
      --tte_index_dir "${index_dir}" \
      --split test \
      --type_vocab_file "data/type_vocab.json" \
      --query_embedding_cache "${query_cache}" \
      --knowledge_encoder_path "${knowledge_encoder}" \
      --knowledge_encoder_base_model_path "${base_model}" \
      --query_max_length 128 \
      --max_table_len 16384 \
      --batch_size "${eval_batch_size}"
  else
    echo "Usage: $0 [train|eval] [all|tacrolimus_abnormal|death]" >&2
    exit 2
  fi
done
