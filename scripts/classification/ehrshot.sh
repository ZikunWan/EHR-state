#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

mode="${1:-train}"
use_eval_dataset="${2:-true}"
task_selector="${3:-all}"
seed_selector="${4:-42}"
data_dir="/data/zikun_workspace/input/tables/ehrshot"
index_dir="/data/zikun_workspace/input/tasks/classification/ehrshot"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/ehrshot/text_embeddings.pt"
checkpoint_root="${CHECKPOINT_ROOT:-/data/zikun_workspace/checkpoints/classification/ehrshot/550M}"
query_cache_dir="/data/zikun_workspace/input/cache/query_embeddings/query_classifier/ehrshot"
format_query_cache="/data/zikun_workspace/input/cache/query_embeddings/pretraining/task_query_knowledge_embeddings.pt"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
pretrained_path="${PRETRAINED_PATH:-/data/zikun_workspace/checkpoints/pretraining/550M}"
model_label="${MODEL_LABEL:-550M}"
train_token_budget="${TRAIN_TOKEN_BUDGET:-32768}"
max_dynamic_train_batch_size="${MAX_DYNAMIC_TRAIN_BATCH_SIZE:-32}"
resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-}"
all_tasks=(guo_los guo_readmission guo_icu lab_anemia lab_hyperkalemia lab_hyponatremia lab_hypoglycemia lab_thrombocytopenia new_acutemi new_celiac new_hyperlipidemia new_hypertension new_lupus new_pancan)
if [[ "${task_selector}" == "all" ]]; then
  tasks=("${all_tasks[@]}")
elif [[ " ${all_tasks[*]} " == *" ${task_selector} "* ]]; then
  tasks=("${task_selector}")
else
  echo "Unknown EHRSHOT task: ${task_selector}" >&2
  echo "Valid tasks: all ${all_tasks[*]}" >&2
  exit 2
fi

if [[ "${seed_selector}" == "all" || "${seed_selector}" == "42-46" ]]; then
  seeds=({42..46})
elif [[ "${seed_selector}" =~ ^[0-9]+$ ]]; then
  seeds=("${seed_selector}")
else
  echo "Invalid seed: ${seed_selector}. Use an integer or 42-46." >&2
  exit 2
fi

if [[ "${mode}" != "train" && "${mode}" != "eval" && "${mode}" != "train_eval" ]]; then
  echo "Usage: $0 [train|eval|train_eval] [true|false] [all|task_name] [seed|42-46]" >&2
  exit 2
fi

for task in "${tasks[@]}"; do
  for seed in "${seeds[@]}"; do
    run_dir="${checkpoint_root}/${task}/seed_${seed}/full_tune"
    metrics_path="${run_dir}/test_metrics_${task}.json"
    if [[ "${mode}" == "train_eval" && -f "${metrics_path}" ]]; then
      echo "Skipping completed ${task} seed=${seed}"
      continue
    fi
    if [[ "${mode}" == "train" || ( "${mode}" == "train_eval" && ! -f "${run_dir}/model.safetensors" ) ]]; then
      resume_args=()
      if [[ "${resume_from_checkpoint}" == "latest" ]]; then
        latest_checkpoint="$(find "${run_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1 || true)"
        if [[ -n "${latest_checkpoint}" ]]; then
          resume_args=(--resume_from_checkpoint "${latest_checkpoint}")
          echo "Resuming ${task} seed=${seed} from ${latest_checkpoint}"
        fi
      elif [[ -n "${resume_from_checkpoint}" ]]; then
        resume_args=(--resume_from_checkpoint "${resume_from_checkpoint}")
      fi
      deepspeed --include localhost:0,1,2,3,4,5,6,7 train/classification/train_encoder_classifier.py \
      --deepspeed "ds_config_zero2.json" \
      --dataset_name ehrshot \
      --data_dir "${data_dir}" \
      --train_info_path "${index_dir}/train/${task}.csv" \
      --val_info_path "${index_dir}/val/${task}.csv" \
      --use_eval_dataset "${use_eval_dataset}" \
      --task_name "${task}" \
      --embedding_cache "${embedding_cache}" \
      --output_dir "${run_dir}" \
      --run_name "ehrshot_${model_label}_${task}_seed${seed}" \
      --type_vocab_file "data/type_vocab.json" \
      --query_embedding_cache "${query_cache_dir}/${task}.pt" \
      --format_query_embedding_cache "${format_query_cache}" \
      --knowledge_encoder_path "${knowledge_encoder}" \
      --knowledge_encoder_base_model_path "${base_model}" \
      --pretrained_path "${pretrained_path}" \
      --classifier_dropout 0.1 \
      --query_max_length 128 \
      --max_table_len 4096 \
      --train_token_budget "${train_token_budget}" \
      --max_dynamic_train_batch_size "${max_dynamic_train_batch_size}" \
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
      --report_to wandb \
      "${resume_args[@]}"
    fi
    if [[ "${mode}" == "eval" || "${mode}" == "train_eval" ]]; then
      torchrun --standalone --nproc_per_node=8 \
      test/classification/test_encoder_classifier.py \
      --dataset_name ehrshot \
      --data_dir "${data_dir}" \
      --sample_info_test_path "${index_dir}/test/${task}.csv" \
      --checkpoint_dir "${run_dir}" \
      --task_name "${task}" \
      --embedding_cache "${embedding_cache}" \
      --type_vocab_file "data/type_vocab.json" \
      --query_embedding_cache "${query_cache_dir}/${task}.pt" \
      --format_query_embedding_cache "${format_query_cache}" \
      --knowledge_encoder_path "${knowledge_encoder}" \
      --knowledge_encoder_base_model_path "${base_model}" \
      --query_max_length 128 \
      --seed "${seed}" \
      --max_table_len 4096 \
      --batch_size 32 \
      --max_tokens_per_batch 262144 \
      --max_dynamic_batch_size 128
    fi
  done
done
