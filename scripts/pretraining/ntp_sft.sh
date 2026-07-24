#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NTP_CHECKPOINT="/data/zikun_workspace/checkpoints/pretraining/550M_mha_ntp"
OUTPUT_DIR="/data/zikun_workspace/checkpoints/pretraining/550M_ntp_sft"

cd "${PROJECT_ROOT}"
bash "${SCRIPT_DIR}/pretrain.sh" \
    --objectives sft \
    --initialization_checkpoint "${NTP_CHECKPOINT}" \
    --max_table_len 4096 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --no_sft_include_tte \
    --sft_recent_token_ratio 0.75 \
    --run_name "550M_ntp_sft" \
    --output_dir "${OUTPUT_DIR}"
