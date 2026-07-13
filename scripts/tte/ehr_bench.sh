#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

mode="${1:-train}"
task_arg="${2:-all}"
data_dir="/data/zikun_workspace/mimic-iv-3.1_tabular"
index_dir="/data/zikun_workspace/input/tasks/time_to_event/mimic_iv/indices"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/mimic_iv/text_embeddings.pt"
checkpoint_root="/data/zikun_workspace/checkpoints/tte/ehr_bench"
query_cache_dir="/data/zikun_workspace/input/cache/query_embeddings/tte/ehr_bench"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
pretrained_path="/data/zikun_workspace/checkpoints/pretraining/1B"
tasks=(Time_to_Inpatient_Mortality_after_ED Time_to_ICU_Transfer_after_ED Time_to_ED_Reattendance Time_to_Hospital_Readmission Time_to_Inpatient_Mortality Time_to_Hospital_Discharge Time_to_ICU_Mortality Time_to_ICU_Discharge Time_to_ICU_Readmission)

if [[ "${task_arg}" != "all" ]]; then
  tasks=("${task_arg}")
fi

for task in "${tasks[@]}"; do
  if [[ "${mode}" == "train" ]]; then
    deepspeed --include localhost:0,1,2,3,4,5,6,7 train/tte/train_ehr_bench_tte.py \
      --deepspeed "ds_config_zero2.json" \
      --data_dir "${data_dir}" \
      --train_info_path "${index_dir}/train/${task}.csv" \
      --val_info_path "${index_dir}/val/${task}.csv" \
      --max_train_samples 3000 \
      --max_eval_samples 1000 \
      --task_name "${task}" \
      --embedding_cache "${embedding_cache}" \
      --output_dir "${checkpoint_root}/${task}" \
      --run_name "ehr_bench_${task}" \
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
    CUDA_VISIBLE_DEVICES=0 python test/tte/test_ehr_bench_tte.py \
      --data_dir "${data_dir}" \
      --sample_info_test_path "${index_dir}/test/${task}.csv" \
      --checkpoint_dir "${checkpoint_root}/${task}" \
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
    echo "Usage: $0 [train|eval] [all|TASK]" >&2
    exit 2
  fi
done
