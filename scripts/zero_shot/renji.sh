#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

data_dir="/data/zikun_workspace/input/tables/renji/raw"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/renji/text_embeddings.pt"
query_cache_dir="/data/zikun_workspace/input/cache/query_embeddings/query_classifier/renji"
pretrained_path="${PRETRAINED_PATH:-/data/zikun_workspace/checkpoints/pretraining/1B}"
output_root="${OUTPUT_ROOT:-/data/zikun_workspace/checkpoints/zero_shot/renji}"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
tasks=(ALB ALP CR Glucose HB INR N_Percent PLT PT TP Uric_Acid WBC)
gpu_ids=(${GPU_IDS:-0 1 2 3 4 5 6 7})
if [[ $# -ge 1 ]]; then
  tasks=("$1")
fi

run_task() {
  local task="$1" gpu="$2"
  TQDM_POSITION="${gpu}" CUDA_VISIBLE_DEVICES="${gpu}" python test/classification/test_encoder_classifier.py \
    --dataset_name renji --data_dir "${data_dir}" --split test \
    --task_name "${task}" --embedding_cache "${embedding_cache}" \
    --type_vocab_file data/type_vocab.json \
    --query_embedding_cache "${query_cache_dir}/${task}.pt" \
    --pretrained_path "${pretrained_path}" --output_dir "${output_root}/${task}" \
    --knowledge_encoder_path "${knowledge_encoder}" \
    --knowledge_encoder_base_model_path "${base_model}" \
    --query_max_length 128 --max_table_len 16384 --batch_size 64
}

for task in "${tasks[@]}"; do
  run_task "${task}" "${gpu_ids[0]}"
done
