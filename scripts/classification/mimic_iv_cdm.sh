#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

mode="${1:-train}"
use_eval_dataset="${2:-true}"
seed="${3:-42}"
data_dir="/data/EHR_data_public/mimic-iv-cdm"
index_dir="/data/zikun_workspace/input/tasks/classification/mimic_iv_cdm"
task="MIMIC-IV-CDM Main Disease Diagnoses"
task_slug="main_disease_diagnoses"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/mimic_iv_cdm/text_embeddings.pt"
checkpoint_root="${CHECKPOINT_ROOT:-/data/zikun_workspace/checkpoints/classification/mimic_iv_cdm/550M_joint_tricks}"
checkpoint_dir="${checkpoint_root}/${task_slug}/seed_${seed}/full_tune"
query_cache="/data/zikun_workspace/input/cache/query_embeddings/query_classifier/mimic_iv_cdm/${task_slug}.pt"
format_query_cache="/data/zikun_workspace/input/cache/query_embeddings/pretraining/task_query_knowledge_embeddings.pt"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
pretrained_path="${PRETRAINED_PATH:-/data/zikun_workspace/checkpoints/pretraining/550M}"

if [[ "${mode}" == "train" || ( "${mode}" == "train_eval" && ! -f "${checkpoint_dir}/model.safetensors" ) ]]; then
  deepspeed --include localhost:0,1,2,3,4,5,6,7 train/classification/train_encoder_classifier.py \
    --deepspeed "ds_config_zero2.json" \
    --dataset_name mimic_iv_cdm \
    --data_dir "${data_dir}" \
    --processed_dir "${index_dir}" \
    --task_name "${task}" \
    --use_eval_dataset "${use_eval_dataset}" \
    --embedding_cache "${embedding_cache}" \
    --output_dir "${checkpoint_dir}" \
    --run_name "mimic_iv_cdm_550M_joint_tricks_${task_slug}_seed${seed}" \
    --type_vocab_file "data/type_vocab.json" \
    --query_embedding_cache "${query_cache}" \
    --format_query_embedding_cache "${format_query_cache}" \
    --knowledge_encoder_path "${knowledge_encoder}" \
    --knowledge_encoder_base_model_path "${base_model}" \
    --pretrained_path "${pretrained_path}" \
    --classifier_dropout 0.1 \
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
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 60 \
    --seed "${seed}" \
    --data_seed "${seed}" \
    --bf16 true \
    --dataloader_num_workers 16 \
    --report_to wandb
fi

if [[ "${mode}" == "eval" || "${mode}" == "train_eval" ]]; then
  torchrun --standalone --nproc_per_node=8 test/classification/test_encoder_classifier.py \
    --dataset_name mimic_iv_cdm \
    --data_dir "${data_dir}" \
    --processed_dir "${index_dir}" \
    --split test \
    --checkpoint_dir "${checkpoint_dir}" \
    --task_name "${task}" \
    --embedding_cache "${embedding_cache}" \
    --type_vocab_file "data/type_vocab.json" \
    --query_embedding_cache "${query_cache}" \
    --format_query_embedding_cache "${format_query_cache}" \
    --knowledge_encoder_path "${knowledge_encoder}" \
    --knowledge_encoder_base_model_path "${base_model}" \
    --query_max_length 128 \
    --seed "${seed}" \
    --max_table_len 4096 \
    --batch_size 32
elif [[ "${mode}" != "train" ]]; then
  echo "Usage: $0 [train|eval|train_eval] [true|false] [seed]" >&2
  exit 2
fi
