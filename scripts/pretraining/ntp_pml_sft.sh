#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="/data/zikun_workspace/checkpoints/pretraining/550M_mha_ntp_pml_sft"
NTP_DIR="${OUTPUT_ROOT}/ntp"
PML_DIR="${OUTPUT_ROOT}/pml"
SFT_DIR="${OUTPUT_ROOT}/sft"

cd "${PROJECT_ROOT}"
bash "${SCRIPT_DIR}/pretrain.sh" \
    --objectives ntp \
    --max_table_len 4096 \
    --run_name "550M_mha_ntp_pml_sft_ntp" \
    --output_dir "${NTP_DIR}"

bash "${SCRIPT_DIR}/pretrain.sh" \
    --objectives pml \
    --initialization_checkpoint "${NTP_DIR}" \
    --max_table_len 4096 \
    --run_name "550M_mha_ntp_pml_sft_pml" \
    --output_dir "${PML_DIR}"

bash "${SCRIPT_DIR}/pretrain.sh" \
    --objectives sft \
    --initialization_checkpoint "${PML_DIR}" \
    --max_table_len 4096 \
    --no_sft_include_tte \
    --sft_recent_token_ratio 0.75 \
    --run_name "550M_mha_ntp_pml_sft_sft" \
    --output_dir "${SFT_DIR}"
