#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

mode="${1:-train}"
data_dir="/data/EHR_data_public/mimic-iv-cdm"
index_dir="/data/zikun_workspace/input/tasks/classification/mimic_iv_cdm"
task="MIMIC-IV-CDM Main Disease Diagnoses"
task_slug="main_disease_diagnoses"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/mimic_iv_cdm/text_embeddings.pt"
checkpoint_dir="/data/zikun_workspace/checkpoints/classification/mimic_iv_cdm/${task_slug}"
query_cache="/data/zikun_workspace/input/cache/query_embeddings/query_classifier/mimic_iv_cdm/${task_slug}.pt"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"

if [[ "${mode}" == "train" ]]; then
  deepspeed --include localhost:0,1,2,3,4,5,6,7 train/classification/train_encoder_classifier.py \
    --deepspeed "ds_config_zero2.json" \
    --dataset_name mimic_iv_cdm \
    --data_dir "${data_dir}" \
    --processed_dir "${index_dir}" \
    --task_name "${task}" \
    --embedding_cache "${embedding_cache}" \
    --output_dir "${checkpoint_dir}" \
    --run_name "mimic_iv_cdm_${task_slug}" \
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
elif [[ "${mode}" == "eval" ]]; then
  CUDA_VISIBLE_DEVICES=0 python test/classification/test_encoder_classifier.py \
    --dataset_name mimic_iv_cdm \
    --data_dir "${data_dir}" \
    --processed_dir "${index_dir}" \
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
else
  echo "Usage: $0 [train|eval]" >&2
  exit 2
fi
