#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

data_dir="/data/zikun_workspace/input/tables/renji/raw"
embedding_cache="/data/zikun_workspace/input/cache/embeddings/renji/text_embeddings.pt"
tte_index_dir="/data/zikun_workspace/input/tasks/time_to_event/renji"
query_cache_dir="/data/zikun_workspace/input/cache/query_embeddings/query_candidate"
pretrained_path="${PRETRAINED_PATH:-/data/zikun_workspace/checkpoints/pretraining/1B}"
output_root="${OUTPUT_ROOT:-/data/zikun_workspace/checkpoints/zero_shot/tte/renji}"
knowledge_encoder="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt"
base_model="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"

# Both Renji TTE tasks now fit the 365-bin pretrained TTE head.  The
# tacrolimus task is piecewise 31/150/185 bins; death uses one 365-day stage.
tasks=(tacrolimus_abnormal death)
gpu_ids=(${GPU_IDS:-0 1 2 3 4 5 6 7})
if [[ $# -ge 1 ]]; then
  tasks=("$1")
fi

run_task() {
    local task="$1" gpu="$2"
    if [[ "${task}" == "death" ]]; then
      query_cache="${query_cache_dir}/renji_death_survival_task_query_knowledge_embeddings.pt"
    else
      query_cache="${query_cache_dir}/renji_survival_task_query_knowledge_embeddings.pt"
    fi
    TQDM_POSITION="${gpu}" CUDA_VISIBLE_DEVICES="${gpu}" python test/tte/test_renji_tte.py \
      --survival_task "${task}" \
      --data_dir "${data_dir}" \
      --embedding_cache "${embedding_cache}" \
      --output_dir "${output_root}/${task}" \
      --patient_subset_path data/patients.json \
      --tte_index_dir "${tte_index_dir}" \
      --split test \
      --type_vocab_file data/type_vocab.json \
      --query_embedding_cache "${query_cache}" \
      --pretrained_path "${pretrained_path}" \
      --knowledge_encoder_path "${knowledge_encoder}" \
      --knowledge_encoder_base_model_path "${base_model}" \
      --query_max_length 128 \
      --max_table_len 16384 \
      --batch_size 128
}

for task in "${tasks[@]}"; do
  run_task "${task}" "${gpu_ids[0]}"
done
