#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

data_dir="/data/zikun_workspace/input/tables/ehrshot"
index_dir="/data/zikun_workspace/input/tasks/classification/ehrshot"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/ehrshot/text_embeddings.pt"
query_cache_dir="/data/zikun_workspace/input/cache/query_embeddings/query_classifier/ehrshot"
format_query_cache="/data/zikun_workspace/input/cache/query_embeddings/pretraining/task_query_knowledge_embeddings.pt"
pretrained_path="/data/zikun_workspace/checkpoints/pretraining/550M"
output_root="/data/zikun_workspace/checkpoints/classification/ehrshot/550M"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [all|task_name]" >&2
  exit 2
fi

task="${1:-all}"
if [[ "${task}" == "all" ]]; then
  torchrun --standalone --nproc_per_node=8 \
    test/classification/test_ehrshot_joint.py \
      --data_dir "${data_dir}" \
      --index_dir "${index_dir}/test" \
      --output_root "${output_root}" \
      --embedding_cache "${embedding_cache}" \
      --query_cache_dir "${query_cache_dir}" \
      --format_query_embedding_cache "${format_query_cache}" \
      --pretrained_path "${pretrained_path}" \
      --type_vocab_file data/type_vocab.json \
      --knowledge_encoder_path "${knowledge_encoder}" \
      --knowledge_encoder_base_model_path "${base_model}" \
      --query_max_length 128 \
      --max_table_len 4096 \
      --max_tokens_per_batch 262144 \
      --max_dynamic_batch_size 128
else
  torchrun --standalone --nproc_per_node=8 \
    test/classification/test_encoder_classifier.py \
      --dataset_name ehrshot \
      --data_dir "${data_dir}" \
      --sample_info_test_path "${index_dir}/test/${task}.csv" \
      --task_name "${task}" \
      --embedding_cache "${embedding_cache}" \
      --type_vocab_file data/type_vocab.json \
      --query_embedding_cache "${query_cache_dir}/${task}.pt" \
      --format_query_embedding_cache "${format_query_cache}" \
      --pretrained_path "${pretrained_path}" \
      --output_dir "${output_root}/${task}/zero_shot" \
      --knowledge_encoder_path "${knowledge_encoder}" \
      --knowledge_encoder_base_model_path "${base_model}" \
      --query_max_length 128 \
      --max_table_len 4096 \
      --batch_size 32 \
      --max_tokens_per_batch 262144 \
      --max_dynamic_batch_size 128
fi
